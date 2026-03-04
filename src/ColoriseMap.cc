#include "colorise.h"
#include <signal.h>
#include <iomanip>
#include "utils.h"
#include <ros/package.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>

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
    fs["base_frame"]              >> base_frame_;
    fs.release();

    // Expand $(find point_cloud_projection) in paths
    std::string pkg = ros::package::getPath("point_cloud_projection");
    auto expandPath = [&](std::string& path) {
        const std::string token = "$(find point_cloud_projection)";
        size_t pos = path.find(token);
        if (pos != std::string::npos) path.replace(pos, token.size(), pkg);
    };
    expandPath(save_path_);
    expandPath(map_pcd_path_);

    // Load pre-built LIO-SAM map PCD (world frame).
    // LIO-SAM saves GlobalMap as plain XYZI — load as PointXYZI then convert.
    map_points_.reset(new pcl::PointCloud<PointXYZRGBIntensity>());
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw(new pcl::PointCloud<pcl::PointXYZI>());
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(map_pcd_path_, *raw) == -1) {
            ROS_ERROR_STREAM("Failed to load map PCD from " << map_pcd_path_);
            ros::shutdown(); return;
        }
        map_points_->points.reserve(raw->points.size());
        for (const auto& p : raw->points) {
            PointXYZRGBIntensity pt;
            pt.x         = p.x;
            pt.y         = p.y;
            pt.z         = p.z;
            pt.intensity = p.intensity;
            pt.rgb       = 0; // will be filled by colorization
            map_points_->points.push_back(pt);
        }
        map_points_->width    = map_points_->points.size();
        map_points_->height   = 1;
        map_points_->is_dense = raw->is_dense;
    }
    ROS_INFO_STREAM("Loaded map: " << map_points_->size() << " points from " << map_pcd_path_);

    // Subscribers
    sub_odom_       = nh_.subscribe(odom_topic_,              200, &PointCloudColorizer::odomCallback,         this);
    sub_img_right_  = nh_.subscribe(image_topic_right_,        10, &PointCloudColorizer::imgRightCallback,     this);
    sub_img_left_   = nh_.subscribe(image_topic_left_,         10, &PointCloudColorizer::imgLeftCallback,      this);
    sub_info_right_ = nh_.subscribe(camera_info_topic_right_,   1, &PointCloudColorizer::camInfoRightCallback, this);
    sub_info_left_  = nh_.subscribe(camera_info_topic_left_,    1, &PointCloudColorizer::camInfoLeftCallback,  this);

    // Publishers
    pub_             = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_,          1);
    pub_raw_map_     = nh_.advertise<sensor_msgs::PointCloud2>("/colorise_map/raw_map", 1, /*latch=*/true);
    pub_progress_    = nh_.advertise<sensor_msgs::PointCloud2>("/colorise_map/progress", 1);
    pub_robot_pose_  = nh_.advertise<geometry_msgs::PoseStamped>("/colorise_map/robot_pose", 1);
    pub_frustum_     = nh_.advertise<visualization_msgs::MarkerArray>("/colorise_map/frustum", 1);

    accumulated_map_.reset(new pcl::PointCloud<PointXYZRGBIntensity>());
    global_instance = this;
    signal(SIGINT, signalHandler);

    color_timer_ = nh_.createTimer(ros::Duration(0.1), &PointCloudColorizer::colorizeFromPCD, this);

    // Publish the grey raw map immediately (latched — RViz will get it on connect)
    publishRawMap();

    ROS_INFO("ColoriseMap running. Raw map published. Waiting for odometry and images...");
}

// ── Publish full uncoloured map in grey ───────────────────────────────────────

void PointCloudColorizer::publishRawMap() {
    pcl::PointCloud<PointXYZRGBIntensity> grey_map = *map_points_;
    for (auto& pt : grey_map.points) {
        uint8_t g = static_cast<uint8_t>(std::min(255.0f, pt.intensity * 0.5f + 80.0f));
        pt.rgb = (static_cast<uint32_t>(g) << 16 |
                  static_cast<uint32_t>(g) << 8  |
                  static_cast<uint32_t>(g));
    }
    grey_map.width    = grey_map.points.size();
    grey_map.height   = 1;
    grey_map.is_dense = false;

    sensor_msgs::PointCloud2 msg;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl::toPCLPointCloud2(grey_map, pcl_pc2);
    pcl_conversions::fromPCL(pcl_pc2, msg);
    msg.header.stamp    = ros::Time::now();
    msg.header.frame_id = "map";
    pub_raw_map_.publish(msg);
    ROS_INFO("[DBG] Published raw map (%zu pts) to /colorise_map/raw_map  subscribers=%d  frame_id=%s",
             grey_map.points.size(), pub_raw_map_.getNumSubscribers(), msg.header.frame_id.c_str());
    if (!grey_map.points.empty()) {
        const auto& p = grey_map.points[0];
        ROS_INFO("[DBG] First map point: x=%.2f y=%.2f z=%.2f intensity=%.2f", p.x, p.y, p.z, p.intensity);
    }
}

// ── CameraInfo callbacks ──────────────────────────────────────────────────────

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
    while (!odom_buffer_.empty() &&
           (msg->header.stamp - odom_buffer_.front()->header.stamp).toSec() > 5.0)
        odom_buffer_.pop_front();
}

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

// ── Publish robot pose as PoseStamped ─────────────────────────────────────────

void PointCloudColorizer::publishRobotPose(const Eigen::Matrix4d& T_world_base, ros::Time t) {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp    = t;
    pose_msg.header.frame_id = "map";

    Eigen::Vector3d    pos = T_world_base.block<3,1>(0,3);
    Eigen::Quaterniond q(T_world_base.block<3,3>(0,0));
    q.normalize();

    pose_msg.pose.position.x    = pos.x();
    pose_msg.pose.position.y    = pos.y();
    pose_msg.pose.position.z    = pos.z();
    pose_msg.pose.orientation.w = q.w();
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();

    pub_robot_pose_.publish(pose_msg);
}

// ── Publish camera frustum as line-list marker ────────────────────────────────

void PointCloudColorizer::publishFrustum(const Eigen::Matrix4d& T_world_camera,
                                         const sensor_msgs::CameraInfo& info,
                                         ros::Time t, int id, float r, float g, float b) {
    double fx = info.K[0], fy = info.K[4];
    double cx = info.K[2], cy = info.K[5];
    double w  = info.width, h = info.height;
    double depth = 3.0;

    auto unproject = [&](double u, double v) -> Eigen::Vector4d {
        return Eigen::Vector4d((u - cx) / fx * depth,
                               (v - cy) / fy * depth,
                               depth, 1.0);
    };

    std::vector<Eigen::Vector4d> corners_cam = {
        unproject(0, 0), unproject(w, 0), unproject(w, h), unproject(0, h),
    };

    Eigen::Vector4d origin_world = T_world_camera * Eigen::Vector4d(0, 0, 0, 1);

    auto toPoint = [&](const Eigen::Vector4d& p_cam) -> geometry_msgs::Point {
        Eigen::Vector4d pw = T_world_camera * p_cam;
        geometry_msgs::Point pt; pt.x = pw.x(); pt.y = pw.y(); pt.z = pw.z(); return pt;
    };
    auto originPoint = [&]() -> geometry_msgs::Point {
        geometry_msgs::Point pt;
        pt.x = origin_world.x(); pt.y = origin_world.y(); pt.z = origin_world.z(); return pt;
    };

    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp    = t;
    marker.ns              = "frustum";
    marker.id              = id;
    marker.type            = visualization_msgs::Marker::LINE_LIST;
    marker.action          = visualization_msgs::Marker::ADD;
    marker.scale.x         = 0.03;
    marker.color.r = r; marker.color.g = g; marker.color.b = b; marker.color.a = 0.8f;
    marker.lifetime        = ros::Duration(0.5);

    for (auto& c : corners_cam) {
        marker.points.push_back(originPoint());
        marker.points.push_back(toPoint(c));
    }
    for (int i = 0; i < 4; ++i) {
        marker.points.push_back(toPoint(corners_cam[i]));
        marker.points.push_back(toPoint(corners_cam[(i+1) % 4]));
    }

    visualization_msgs::MarkerArray arr;
    arr.markers.push_back(marker);
    pub_frustum_.publish(arr);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

void PointCloudColorizer::saveFinalMap() {
    if (!accumulated_map_ || accumulated_map_->points.empty()) {
        ROS_WARN("No coloured points accumulated yet — skipping save.");
        return;
    }
    accumulated_map_->width    = accumulated_map_->points.size();
    accumulated_map_->height   = 1;
    accumulated_map_->is_dense = false;
    ROS_INFO("Saving final coloured map (%zu points) to %s",
             accumulated_map_->points.size(), save_path_.c_str());
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

void PointCloudColorizer::colorizeFromPCD(const ros::TimerEvent&) {

    // ── Prerequisites ─────────────────────────────────────────────────────────
    ROS_INFO_THROTTLE(3.0,
        "[ColoriseMap] cam_R=%s cam_L=%s map=%zu img_R=%zu img_L=%zu odom=%zu "
        "base_frame='%s' accumulated=%zu",
        cam_info_right_ ? "OK" : "MISSING",
        cam_info_left_  ? "OK" : "MISSING",
        map_points_ ? map_points_->size() : 0ul,
        img_right_buffer_.size(),
        img_left_buffer_.size(),
        odom_buffer_.size(),
        base_frame_.c_str(),
        accumulated_map_ ? accumulated_map_->size() : 0ul);

    if (!cam_info_right_ || !cam_info_left_) {
        ROS_WARN_THROTTLE(5.0, "[ColoriseMap] Waiting for camera_info: %s  %s",
                          camera_info_topic_right_.c_str(), camera_info_topic_left_.c_str());
        return;
    }
    if (!map_points_ || map_points_->empty()) return;
    if (img_right_buffer_.empty() || img_left_buffer_.empty()) {
        ROS_WARN_THROTTLE(3.0, "[ColoriseMap] image buffers empty — topics: %s  %s",
                          image_topic_right_.c_str(), image_topic_left_.c_str());
        return;
    }
    if (odom_buffer_.empty()) {
        ROS_WARN_THROTTLE(2.0, "[ColoriseMap] odom buffer empty — topic: %s", odom_topic_.c_str());
        return;
    }

    // Fallback: if base_frame_ not set in config, derive from odometry message
    if (base_frame_.empty()) {
        base_frame_ = odom_buffer_.back()->child_frame_id;
        ROS_WARN_ONCE("[ColoriseMap] base_frame not set in config — using odom child_frame_id: '%s'. "
                      "Add 'base_frame: <your_base_frame>' to config.yaml.",
                      base_frame_.c_str());
    }

    // ── Match images ──────────────────────────────────────────────────────────
    ros::Time t_img = img_right_buffer_.front()->header.stamp;
    auto img_right_msg = findClosest(img_right_buffer_, t_img);
    auto img_left_msg  = findClosest(img_left_buffer_,  t_img);
    img_right_buffer_.pop_front();
    img_left_buffer_.pop_front();

    if (!img_right_msg || !img_left_msg) {
        ROS_WARN_THROTTLE(2.0, "[ColoriseMap] No matching image pair at t=%.3f "
                          "(max_time_offset=%.3f)", t_img.toSec(), max_time_offset_);
        return;
    }

    // ── Odometry ──────────────────────────────────────────────────────────────
    Eigen::Matrix4d T_world_base;
    if (!interpolateOdometry(t_img, T_world_base)) {
        ROS_WARN_THROTTLE(2.0, "[ColoriseMap] interpolateOdometry failed t=%.3f "
                          "odom=[%.3f, %.3f]",
                          t_img.toSec(),
                          odom_buffer_.front()->header.stamp.toSec(),
                          odom_buffer_.back()->header.stamp.toSec());
        return;
    }

    // ── TF extrinsics ─────────────────────────────────────────────────────────
    auto lookupTBaseCamera = [&](const std::string& cam_frame,
                                 Eigen::Matrix4d& T_out) -> bool {
        try {
            T_out = tf2::transformToEigen(
                tf_buffer_.lookupTransform(base_frame_, cam_frame,
                                           ros::Time(0), ros::Duration(0.1))).matrix();
            return true;
        } catch (const tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(2.0, "[ColoriseMap] TF '%s'->'%s' failed: %s",
                              base_frame_.c_str(), cam_frame.c_str(), ex.what());
            return false;
        }
    };

    Eigen::Matrix4d T_base_right, T_base_left;
    if (!lookupTBaseCamera(cam_info_right_->header.frame_id, T_base_right) ||
        !lookupTBaseCamera(cam_info_left_->header.frame_id,  T_base_left))
        return;

    // World pose of each camera, then invert:
    // map points are in world frame → T_cam_from_world = T_world_cam^-1
    Eigen::Matrix4d T_world_right      = T_world_base * T_base_right;
    Eigen::Matrix4d T_world_left       = T_world_base * T_base_left;
    Eigen::Matrix4d T_right_from_world = T_world_right.inverse();
    Eigen::Matrix4d T_left_from_world  = T_world_left.inverse();

    ROS_DEBUG_THROTTLE(1.0, "[ColoriseMap] Robot: (%.2f,%.2f,%.2f)  "
                       "RightCam: (%.2f,%.2f,%.2f)",
                       T_world_base(0,3),  T_world_base(1,3),  T_world_base(2,3),
                       T_world_right(0,3), T_world_right(1,3), T_world_right(2,3));

    // ── Decode images ─────────────────────────────────────────────────────────
    cv::Mat img_right = cv::imdecode(cv::Mat(img_right_msg->data), cv::IMREAD_COLOR);
    cv::Mat img_left  = cv::imdecode(cv::Mat(img_left_msg->data),  cv::IMREAD_COLOR);
    if (img_right.empty() || img_left.empty()) {
        ROS_WARN_THROTTLE(2.0, "[ColoriseMap] Image decode failed");
        return;
    }

    auto buildK = [](const sensor_msgs::CameraInfo& info) {
        cv::Mat_<double> K(3, 3);
        K << info.K[0], 0.0, info.K[2], 0.0, info.K[4], info.K[5], 0.0, 0.0, 1.0;
        return K;
    };
    auto buildDist = [](const sensor_msgs::CameraInfo& info) {
        cv::Mat d(info.D, true); return d.reshape(1, 1);
    };

    // ── Project and colorize ──────────────────────────────────────────────────
    std::vector<cv::Point3f> P3;
    P3.reserve(map_points_->points.size());
    for (const auto& pt : map_points_->points)
        P3.emplace_back(pt.x, pt.y, pt.z);

    pcl::PointCloud<PointXYZRGBIntensity>::Ptr temp_colored(
        new pcl::PointCloud<PointXYZRGBIntensity>());

    colorize(P3, img_right, T_right_from_world,
             buildK(*cam_info_right_), buildDist(*cam_info_right_),
             cam_info_right_->distortion_model,
             (int)cam_info_right_->width, (int)cam_info_right_->height,
             temp_colored, map_points_);

    colorize(P3, img_left, T_left_from_world,
             buildK(*cam_info_left_), buildDist(*cam_info_left_),
             cam_info_left_->distortion_model,
             (int)cam_info_left_->width, (int)cam_info_left_->height,
             temp_colored, map_points_);

    ROS_INFO_THROTTLE(1.0, "[ColoriseMap] t=%.2f: %zu points colored this frame",
                      t_img.toSec(), temp_colored->size());

    if (temp_colored->empty()) {
        ROS_WARN_THROTTLE(2.0, "[ColoriseMap] 0 points colored — "
                          "check base_frame ('%s'), TF tree, and map/odom alignment. "
                          "RightCam world pos: (%.2f,%.2f,%.2f)",
                          base_frame_.c_str(),
                          T_world_right(0,3), T_world_right(1,3), T_world_right(2,3));
        return;
    }

    // ── Publish live feedback ─────────────────────────────────────────────────
    publishRobotPose(T_world_base, t_img);
    publishFrustum(T_world_right, *cam_info_right_, t_img, 0, 0.2f, 0.8f, 0.2f);
    publishFrustum(T_world_left,  *cam_info_left_,  t_img, 1, 0.2f, 0.2f, 0.8f);

    // Progress topic: always publish this frame's colored points immediately,
    // regardless of min_color_frames_. Subscribe to /colorise_map/progress in
    // RViz for immediate visual feedback.
    {
        sensor_msgs::PointCloud2 progress_msg;
        pcl::PCLPointCloud2 tmp;
        pcl::toPCLPointCloud2(*temp_colored, tmp);
        pcl_conversions::fromPCL(tmp, progress_msg);
        progress_msg.header.stamp    = t_img;
        progress_msg.header.frame_id = "map";
        pub_progress_.publish(progress_msg);
    }

    // ── Accumulate colors ─────────────────────────────────────────────────────
    // min_color_frames_ == 1: skip history, add every colored point immediately.
    // min_color_frames_ > 1:  average over N observations (noise reduction).
    if (min_color_frames_ <= 1) {
        for (const auto& pt : temp_colored->points)
            accumulated_map_->points.push_back(pt);
    } else {
        for (const auto& pt : temp_colored->points) {
            PointKey key{pt.x, pt.y, pt.z};
            color_history_[key].push_back(rgbToVec3b(pt.rgb));
            if ((int)color_history_[key].size() >= min_color_frames_) {
                PointXYZRGBIntensity out_pt = pt;
                out_pt.rgb = vec3bToRgb(averageColor(color_history_[key]));
                accumulated_map_->points.push_back(out_pt);
                color_history_.erase(key);
            }
        }
    }

    size_t total    = map_points_->size();
    size_t coloured = accumulated_map_->points.size();
    ROS_INFO_STREAM_THROTTLE(2.0, "[ColoriseMap] Accumulated: " << coloured
                             << " / " << total << " ("
                             << std::fixed << std::setprecision(1)
                             << (100.0 * coloured / total) << "%)  "
                             << "pending in history: " << color_history_.size());

    // ── Publish accumulated map ───────────────────────────────────────────────
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

// ── colorize() ────────────────────────────────────────────────────────────────
//
// T_cam_from_points: transforms points into camera frame directly.
// For ColoriseScan: computed by computeCamFromLidar / computeCamFromLidarIMU.
// For ColoriseMap:  T_world_cam.inverse() (points are already in world frame).
// No internal inversion — the caller is responsible for passing the correct
// camera-from-points transform.

void PointCloudColorizer::colorize(const std::vector<cv::Point3f>& P3,
                                   const cv::Mat& img,
                                   const Eigen::Matrix4d& T_cam_from_points,
                                   const cv::Mat& K,
                                   const cv::Mat& dist,
                                   const std::string& distortion_model,
                                   int width, int height,
                                   pcl::PointCloud<PointXYZRGBIntensity>::Ptr& out,
                                   const pcl::PointCloud<PointXYZRGBIntensity>::Ptr& in,
                                   float source_z_max)
{
    Eigen::Matrix3d R_e = T_cam_from_points.block<3,3>(0,0);
    Eigen::Vector3d t_e = T_cam_from_points.block<3,1>(0,3);
    cv::Mat R_cv, rvec, tvec(3, 1, CV_64F);
    cv::eigen2cv(R_e, R_cv);
    cv::Rodrigues(R_cv, rvec);
    for (int i = 0; i < 3; ++i) tvec.at<double>(i, 0) = t_e(i);

    std::vector<cv::Point2f> P2;
    if (distortion_model == "equidistant")
        cv::fisheye::projectPoints(P3, P2, rvec, tvec, K, dist);
    else
        cv::projectPoints(P3, P2, rvec, tvec, K, dist);

    for (size_t i = 0; i < P2.size(); ++i) {
        // source_z_max filter: meaningful for lidar-frame scan colorization,
        // pass FLT_MAX (default) for map colorization where Z is world height.
        if (P3[i].z > source_z_max) continue;

        // Depth check in camera frame — rejects points behind the camera
        Eigen::Vector4d pt_cam = T_cam_from_points *
                                 Eigen::Vector4d(P3[i].x, P3[i].y, P3[i].z, 1.0);
        if (pt_cam.z() <= 0) continue;

        int u = static_cast<int>(std::round(P2[i].x));
        int v = static_cast<int>(std::round(P2[i].y));
        if (u < 0 || u >= width || v < 0 || v >= height) continue;

        cv::Vec3b c = img.at<cv::Vec3b>(v, u);
        if (c[0] < 5 && c[1] < 5 && c[2] < 5) continue; // skip near-black pixels

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

// ── Stubs (not used by map node) ──────────────────────────────────────────────

void PointCloudColorizer::cloudCallback(const sensor_msgs::PointCloud2ConstPtr&) {}
void PointCloudColorizer::imuCallback(const sensor_msgs::ImuConstPtr&) {}
void PointCloudColorizer::trySyncAndProcess() {}
bool PointCloudColorizer::lookupPose(const ros::Time&, Eigen::Matrix4d&) { return false; }
void PointCloudColorizer::pathCallback(const nav_msgs::PathConstPtr&) {}
void PointCloudColorizer::mapCallback(const sensor_msgs::PointCloud2ConstPtr&) {}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    ros::init(argc, argv, "colorise_map_node");
    ros::NodeHandle nh("~");

    std::string pkg = ros::package::getPath("point_cloud_projection");
    std::string default_config = pkg + "/configs/config.yaml";
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