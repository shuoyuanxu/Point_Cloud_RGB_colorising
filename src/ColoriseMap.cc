#include "colorise.h"
#include <signal.h>
#include "utils.h"

static PointCloudColorizer* global_instance = nullptr;

void signalHandler(int sig) {
    if (global_instance) global_instance->saveFinalMap();
    ros::shutdown();
}

PointCloudColorizer::PointCloudColorizer(ros::NodeHandle& nh)
    : nh_(nh), tf_listener_(tf_buffer_)
{
    nh_.param("config_path", config_path_, std::string("../configs/config.yaml"));

    cv::FileStorage fs(config_path_, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        ROS_ERROR("Unable to open config file: %s", config_path_.c_str());
        ros::shutdown(); return;
    }

    fs["max_time_offset"]         >> max_time_offset_;
    fs["initial_startup_delay"]   >> initial_startup_delay_;
    fs["image_topic_right"]       >> image_topic_right_;
    fs["image_topic_left"]        >> image_topic_left_;
    fs["output_topic"]            >> output_topic_;
    fs["odom_topic"]              >> odom_topic_;
    fs["min_color_frames"]        >> min_color_frames_;
    fs["save_pcd_path"]           >> save_path_;
    fs["map_pcd_path"]            >> map_pcd_path_;
    fs["lidar_frame"]             >> lidar_frame_;
    fs["camera_info_topic_right"] >> camera_info_topic_right_;
    fs["camera_info_topic_left"]  >> camera_info_topic_left_;
    fs.release();

    std::string pkg = ros::package::getPath("point_cloud_projection");
    auto expandPath = [&](std::string& path) {
    const std::string token = "$(find point_cloud_projection)";
    size_t pos = path.find(token);
    if (pos != std::string::npos)
        path.replace(pos, token.size(), pkg);
    };

    expandPath(save_path_);
    expandPath(map_pcd_path_);

    // Load pre-built LIO-SAM map PCD (points are in world/map frame)
    map_points_.reset(new pcl::PointCloud<PointXYZRGBIntensity>());
    if (pcl::io::loadPCDFile<PointXYZRGBIntensity>(map_pcd_path_, *map_points_) == -1) {
        ROS_ERROR_STREAM("Failed to load map PCD from " << map_pcd_path_);
        ros::shutdown(); return;
    }
    ROS_INFO_STREAM("Loaded map: " << map_points_->size() << " points from " << map_pcd_path_);

    // Subscribers
    sub_odom_       = nh_.subscribe(odom_topic_,           200, &PointCloudColorizer::odomCallback,        this);
    sub_img_right_  = nh_.subscribe(image_topic_right_,     10, &PointCloudColorizer::imgRightCallback,    this);
    sub_img_left_   = nh_.subscribe(image_topic_left_,      10, &PointCloudColorizer::imgLeftCallback,     this);
    sub_info_right_ = nh_.subscribe(camera_info_topic_right_, 1, &PointCloudColorizer::camInfoRightCallback, this);
    sub_info_left_  = nh_.subscribe(camera_info_topic_left_,  1, &PointCloudColorizer::camInfoLeftCallback,  this);

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);

    accumulated_map_.reset(new pcl::PointCloud<PointXYZRGBIntensity>());
    global_instance = this;
    signal(SIGINT, signalHandler);

    color_timer_ = nh_.createTimer(ros::Duration(0.1), &PointCloudColorizer::colorizeFromPCD, this);
    ROS_INFO("ColoriseMap running. Waiting for odometry and images...");
}

// ── CameraInfo callbacks ─────────────────────────────────────────────────────

void PointCloudColorizer::camInfoRightCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    if (!cam_info_right_) {
        cam_info_right_ = *msg;
        ROS_INFO("Right camera info (frame: %s, %dx%d)",
                 msg->header.frame_id.c_str(), msg->width, msg->height);
    }
}

void PointCloudColorizer::camInfoLeftCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    if (!cam_info_left_) {
        cam_info_left_ = *msg;
        ROS_INFO("Left camera info (frame: %s, %dx%d)",
                 msg->header.frame_id.c_str(), msg->width, msg->height);
    }
}

// ── Odometry buffer ───────────────────────────────────────────────────────────

void PointCloudColorizer::odomCallback(const nav_msgs::OdometryConstPtr& msg) {
    odom_buffer_.push_back(msg);
    // Keep a 5-second window (map node is slower so keep more history)
    while (!odom_buffer_.empty() &&
           (msg->header.stamp - odom_buffer_.front()->header.stamp).toSec() > 5.0)
        odom_buffer_.pop_front();
}

// Interpolate odometry at time t -> T_world_base (4x4 pose in world frame)
bool PointCloudColorizer::interpolateOdometry(ros::Time t, Eigen::Matrix4d& T_out) {
    if (odom_buffer_.size() < 2) return false;

    if (t >= odom_buffer_.back()->header.stamp) {
        const auto& p = odom_buffer_.back()->pose.pose;
        Eigen::Quaterniond q(p.orientation.w, p.orientation.x,
                             p.orientation.y, p.orientation.z);
        T_out = Eigen::Matrix4d::Identity();
        T_out.block<3,3>(0,0) = q.normalized().toRotationMatrix();
        T_out(0,3) = p.position.x;
        T_out(1,3) = p.position.y;
        T_out(2,3) = p.position.z;
        return true;
    }

    for (size_t i = 0; i + 1 < odom_buffer_.size(); ++i) {
        if (odom_buffer_[i]->header.stamp   <= t &&
            odom_buffer_[i+1]->header.stamp >= t) {
            double dt_total = (odom_buffer_[i+1]->header.stamp - odom_buffer_[i]->header.stamp).toSec();
            double dt_query = (t - odom_buffer_[i]->header.stamp).toSec();
            double alpha    = (dt_total > 1e-9) ? dt_query / dt_total : 0.0;

            const auto& p0 = odom_buffer_[i]->pose.pose;
            const auto& p1 = odom_buffer_[i+1]->pose.pose;

            Eigen::Vector3d t0(p0.position.x, p0.position.y, p0.position.z);
            Eigen::Vector3d t1(p1.position.x, p1.position.y, p1.position.z);
            Eigen::Quaterniond q0(p0.orientation.w, p0.orientation.x,
                                  p0.orientation.y, p0.orientation.z);
            Eigen::Quaterniond q1(p1.orientation.w, p1.orientation.x,
                                  p1.orientation.y, p1.orientation.z);

            T_out = Eigen::Matrix4d::Identity();
            T_out.block<3,3>(0,0) = q0.normalized().slerp(alpha, q1.normalized()).toRotationMatrix();
            T_out.block<3,1>(0,3) = t0 + alpha * (t1 - t0);
            return true;
        }
    }
    return false;
}

// ── Image buffer callbacks ────────────────────────────────────────────────────

void PointCloudColorizer::imgRightCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
    img_right_buffer_.push_back(msg);
    cleanOldMsgs(img_right_buffer_, msg->header.stamp);
}

void PointCloudColorizer::imgLeftCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
    img_left_buffer_.push_back(msg);
    cleanOldMsgs(img_left_buffer_, msg->header.stamp);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

void PointCloudColorizer::saveFinalMap() {
    ROS_INFO("Saving final coloured map (%zu points) to %s",
             accumulated_map_->points.size(), save_path_.c_str());
    accumulated_map_->width    = accumulated_map_->points.size();
    accumulated_map_->height   = 1;
    accumulated_map_->is_dense = false;
    pcl::io::savePCDFileBinary(save_path_, *accumulated_map_);
    ROS_INFO("Saved.");
}

cv::Vec3b PointCloudColorizer::rgbToVec3b(uint32_t rgb) {
    return cv::Vec3b(rgb & 0xFF, (rgb >> 8) & 0xFF, (rgb >> 16) & 0xFF);
}
uint32_t PointCloudColorizer::vec3bToRgb(const cv::Vec3b& c) {
    return (static_cast<uint32_t>(c[2]) << 16 |
            static_cast<uint32_t>(c[1]) << 8  |
            static_cast<uint32_t>(c[0]));
}
cv::Vec3b PointCloudColorizer::averageColor(const std::vector<cv::Vec3b>& colors) {
    cv::Vec3i sum(0, 0, 0);
    for (const auto& c : colors) sum += c;
    sum /= static_cast<int>(colors.size());
    return cv::Vec3b(sum[0], sum[1], sum[2]);
}

template<typename T>
void PointCloudColorizer::cleanOldMsgs(std::deque<T>& buffer, ros::Time latest_time) {
    while (!buffer.empty() && (latest_time - buffer.front()->header.stamp).toSec() > 2.0)
        buffer.pop_front();
}

template<typename T>
typename T::value_type PointCloudColorizer::findClosest(const T& buffer, ros::Time target_time) {
    typename T::value_type best_match = nullptr;
    double best_diff = std::numeric_limits<double>::max();
    for (const auto& msg : buffer) {
        double diff = fabs((msg->header.stamp - target_time).toSec());
        if (diff < best_diff && diff <= max_time_offset_)
            best_match = msg, best_diff = diff;
    }
    return best_match;
}

// ── Main colorization timer ───────────────────────────────────────────────────
//
// For each image frame:
//   1. Find closest odometry pose -> T_world_base (robot position in world frame)
//   2. Look up static T_base_camera from TF
//   3. Compute T_world_camera = T_world_base * T_base_camera
//   4. Project all world-frame map points through T_camera_world = T_world_camera^-1
//   5. Colour points that land inside the image
//   6. Accumulate colours across frames and publish

void PointCloudColorizer::colorizeFromPCD(const ros::TimerEvent&) {
    if (!cam_info_right_ || !cam_info_left_) {
        ROS_WARN_THROTTLE(5.0, "Waiting for camera_info...");
        return;
    }
    if (!map_points_ || map_points_->empty()) return;
    if (img_right_buffer_.empty() || img_left_buffer_.empty()) return;
    if (odom_buffer_.empty()) {
        ROS_WARN_THROTTLE(2.0, "Waiting for odometry...");
        return;
    }

    // Use the oldest unprocessed image as the reference time
    ros::Time t_img = img_right_buffer_.front()->header.stamp;

    auto img_right_msg = findClosest(img_right_buffer_, t_img);
    auto img_left_msg  = findClosest(img_left_buffer_,  t_img);
    if (!img_right_msg || !img_left_msg) return;

    // Get robot pose in world frame at image time
    Eigen::Matrix4d T_world_base;
    if (!interpolateOdometry(t_img, T_world_base)) {
        ROS_WARN_THROTTLE(2.0, "No odometry for image time %.3f", t_img.toSec());
        img_right_buffer_.pop_front();
        img_left_buffer_.pop_front();
        return;
    }

    // Build camera intrinsics
    auto buildK = [](const sensor_msgs::CameraInfo& info) {
        cv::Mat_<double> K(3, 3);
        K << info.K[0], 0.0,       info.K[2],
             0.0,       info.K[4], info.K[5],
             0.0,       0.0,       1.0;
        return K;
    };
    auto buildDist = [](const sensor_msgs::CameraInfo& info) {
        cv::Mat d(info.D, /*copyData=*/true);
        return d.reshape(1, 1); // 1xN for cv::fisheye
    };

    // Look up static T_base_camera from TF
    // Returns T_base_camera: transforms camera-frame points to base_link frame
    auto lookupTBaseCamera = [&](const std::string& camera_frame) -> Eigen::Matrix4d {
        try {
            // child_frame_id of odometry message is base_link (or equivalent)
            const std::string& base_frame = odom_buffer_.back()->child_frame_id;
            geometry_msgs::TransformStamped ts =
                tf_buffer_.lookupTransform(base_frame, camera_frame,
                                           ros::Time(0), ros::Duration(0.1));
            return tf2::transformToEigen(ts).matrix();
        } catch (const tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(2.0, "TF base<-camera failed: %s", ex.what());
            return Eigen::Matrix4d::Identity();
        }
    };

    Eigen::Matrix4d T_base_right  = lookupTBaseCamera(cam_info_right_->header.frame_id);
    Eigen::Matrix4d T_base_left   = lookupTBaseCamera(cam_info_left_->header.frame_id);

    // T_world_camera = T_world_base * T_base_camera
    // colorize() inverts this internally to get T_camera_world for projection
    Eigen::Matrix4d T_world_right = T_world_base * T_base_right;
    Eigen::Matrix4d T_world_left  = T_world_base * T_base_left;

    cv::Mat img_right = cv::imdecode(cv::Mat(img_right_msg->data), cv::IMREAD_COLOR);
    cv::Mat img_left  = cv::imdecode(cv::Mat(img_left_msg->data),  cv::IMREAD_COLOR);

    cv::Mat K_right    = buildK(*cam_info_right_);
    cv::Mat K_left     = buildK(*cam_info_left_);
    cv::Mat dist_right = buildDist(*cam_info_right_);
    cv::Mat dist_left  = buildDist(*cam_info_left_);

    // Build world-frame point list
    std::vector<cv::Point3f> P3;
    P3.reserve(map_points_->points.size());
    for (const auto& pt : map_points_->points)
        P3.emplace_back(pt.x, pt.y, pt.z);

    pcl::PointCloud<PointXYZRGBIntensity>::Ptr temp_colored(new pcl::PointCloud<PointXYZRGBIntensity>());

    colorize(P3, img_right, T_world_right, K_right, dist_right,
             cam_info_right_->distortion_model,
             (int)cam_info_right_->width, (int)cam_info_right_->height,
             true, false, temp_colored, map_points_);

    colorize(P3, img_left, T_world_left, K_left, dist_left,
             cam_info_left_->distortion_model,
             (int)cam_info_left_->width, (int)cam_info_left_->height,
             false, true, temp_colored, map_points_);

    // Pop the images we just processed
    img_right_buffer_.pop_front();
    img_left_buffer_.pop_front();

    if (temp_colored->empty()) {
        ROS_WARN_THROTTLE(2.0, "No points coloured in this frame");
        return;
    }

    // Accumulate colours per point — average across min_color_frames_ observations
    for (const auto& pt : temp_colored->points) {
        PointKey key{pt.x, pt.y, pt.z};
        color_history_[key].push_back(rgbToVec3b(pt.rgb));

        if ((int)color_history_[key].size() >= min_color_frames_) {
            cv::Vec3b color = averageColor(color_history_[key]);
            PointXYZRGBIntensity out_pt = pt;
            out_pt.rgb = vec3bToRgb(color);
            accumulated_map_->points.push_back(out_pt);
            color_history_.erase(key);
        }
    }

    ROS_INFO_STREAM_THROTTLE(2.0, "Accumulated: " << accumulated_map_->points.size()
                             << " / " << map_points_->size() << " points coloured");

    // Publish accumulated map
    accumulated_map_->width    = accumulated_map_->points.size();
    accumulated_map_->height   = 1;
    accumulated_map_->is_dense = false;

    sensor_msgs::PointCloud2 out_msg;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl::toPCLPointCloud2(*accumulated_map_, pcl_pc2);
    pcl_conversions::fromPCL(pcl_pc2, out_msg);
    out_msg.header.stamp    = t_img;
    out_msg.header.frame_id = "map";
    pub_.publish(out_msg);
}

// ── colorize(): project world-frame points into camera and colour them ────────
//
// T_camera_lidar here is actually T_world_camera (naming inherited from scan node).
// It is inverted internally to get T_camera_world, which transforms world points
// into camera frame for projection.

void PointCloudColorizer::colorize(const std::vector<cv::Point3f>& P3, const cv::Mat& img,
                                   const Eigen::Matrix4d& T_camera_lidar, const cv::Mat& K, const cv::Mat& dist,
                                   const std::string& distortion_model, int width, int height,
                                   bool is_right, bool mirror_u,
                                   pcl::PointCloud<PointXYZRGBIntensity>::Ptr& out,
                                   const pcl::PointCloud<PointXYZRGBIntensity>::Ptr& in)
{
    // Invert: T_camera_world transforms world-frame points into camera frame
    Eigen::Matrix4d T = T_camera_lidar.inverse();
    Eigen::Matrix3d R_e = T.block<3,3>(0,0);
    Eigen::Vector3d t_e = T.block<3,1>(0,3);

    cv::Mat R_cv, rvec, tvec(3,1,CV_64F);
    cv::eigen2cv(R_e, R_cv);
    cv::Rodrigues(R_cv, rvec);
    for (int i = 0; i < 3; ++i) tvec.at<double>(i,0) = t_e(i);

    std::vector<cv::Point2f> P2;
    if (distortion_model == "equidistant")
        cv::fisheye::projectPoints(P3, P2, rvec, tvec, K, dist);
    else
        cv::projectPoints(P3, P2, rvec, tvec, K, dist);

    for (size_t i = 0; i < P2.size(); ++i) {
        // Only colour points in front of the camera
        Eigen::Vector4d pt_cam = T * Eigen::Vector4d(P3[i].x, P3[i].y, P3[i].z, 1.0);
        if (pt_cam.z() <= 0) continue;

        int u = static_cast<int>(std::round(P2[i].x));
        int v = static_cast<int>(std::round(P2[i].y));
        if (u < 0 || u >= width || v < 0 || v >= height) continue;

        cv::Vec3b c = img.at<cv::Vec3b>(v, u);
        // Skip near-black pixels (likely sky or lens flare artefacts)
        if (c[0] < 5 && c[1] < 5 && c[2] < 5) continue;

        PointXYZRGBIntensity pt;
        pt.x         = P3[i].x;
        pt.y         = P3[i].y;
        pt.z         = P3[i].z;
        pt.intensity = in->points[i].intensity;
        pt.rgb       = (static_cast<uint32_t>(c[2]) << 16 |
                        static_cast<uint32_t>(c[1]) << 8  |
                        static_cast<uint32_t>(c[0]));
        out->points.push_back(pt);
    }
}

// ── Stubs required by header (not used in map node) ──────────────────────────

void PointCloudColorizer::cloudCallback(const sensor_msgs::PointCloud2ConstPtr&) {}
void PointCloudColorizer::imuCallback(const sensor_msgs::ImuConstPtr&) {}
void PointCloudColorizer::trySyncAndProcess() {}
bool PointCloudColorizer::lookupPose(const ros::Time&, Eigen::Matrix4d&) { return false; }
void PointCloudColorizer::pathCallback(const nav_msgs::PathConstPtr&) {}
void PointCloudColorizer::mapCallback(const sensor_msgs::PointCloud2ConstPtr&) {}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    ros::init(argc, argv, "colorize_pointcloud_node"); // or colorise_map_node
    ros::NodeHandle nh("~");
    // Build default config path relative to package
    std::string package_path = ros::package::getPath("point_cloud_projection");
    std::string default_config = package_path + "/configs/config.yaml";
    std::string config_path;
    nh.param("config_path", config_path, default_config);
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    double startup_delay = 0.5;
    if (fs.isOpened()) fs["initial_startup_delay"] >> startup_delay;
    fs.release();

    ROS_INFO("Waiting %.2f sec for sensor startup...", startup_delay);
    ros::Duration(startup_delay).sleep();

    PointCloudColorizer node(nh);
    ros::spin();
    return 0;
}