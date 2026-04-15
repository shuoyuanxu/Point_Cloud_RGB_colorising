#include "colorise.h"
#include "utils.h"

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
    fs["pointcloud_topic"]        >> cloud_topic_;
    fs["output_topic"]            >> output_topic_;
    fs["keep_uncolored_points"]   >> keep_uncolored_points_;
    fs["max_lidar_z"]             >> max_lidar_z_;
    fs["lidar_frame"]             >> lidar_frame_;
    fs["camera_info_topic_right"] >> camera_info_topic_right_;
    fs["camera_info_topic_left"]  >> camera_info_topic_left_;
    fs["imu_topic"]               >> imu_topic_;
    fs["imu_frame"]               >> imu_frame_;
    std::string use_calib_str = "true";
    fs["use_calibration_file"] >> use_calib_str;
    bool use_calibration_file = (use_calib_str != "false" && use_calib_str != "0");
    fs.release();

    // ── Extrinsics: hardcoded from calibration.yaml OR live TF lookup ─────────
    if (use_calibration_file) {
        // Values from configs/calibration.yaml
        //
        // T_forwardLeft_cam_os_sensor  (cam_left from lidar, direct)
        T_cam_lidar_left_ <<
             0.0365235158, -0.9992455766, -0.0132026663,  0.0715811084,
             0.0007068971,  0.0132373111, -0.9999121331, -0.0815248470,
             0.9993325438,  0.0365109736,  0.0011898369, -0.1028095976,
             0.0,           0.0,           0.0,           1.0;

        // T_os_sensor_forwardRight_cam  (lidar from cam_right) — inverted to get cam_right from lidar
        Eigen::Matrix4d T_lidar_cam_right;
        T_lidar_cam_right <<
             0.0124443801, -0.0001078424,  0.9999225599,  0.1023230120,
            -0.9998382680,  0.0129833120,  0.0124447313, -0.0631096435,
            -0.0129836487, -0.9999157074,  0.0000537442, -0.0825364298,
             0.0,           0.0,           0.0,           1.0;
        T_cam_lidar_right_ = T_lidar_cam_right.inverse();

        // T_imu_link_os_sensor  (rotation block only, imu from lidar)
        R_imu_lidar_ <<
             0.9998459674, -0.0157202487,  0.0078048180,
             0.0155436449,  0.9996328268,  0.0221947415,
            -0.0081508592, -0.0220700073,  0.9997232008;

        calib_loaded_ = true;
        ROS_INFO("[ColoriseScan] Extrinsics loaded from calibration.yaml constants");
    } else {
        ROS_INFO("[ColoriseScan] Extrinsics will be looked up from TF tree at runtime");
    }

    // ── Subscribers ───────────────────────────────────────────────────────────
    sub_cloud_      = nh_.subscribe(cloud_topic_,            10,  &PointCloudColorizer::cloudCallback,        this);
    sub_img_right_  = nh_.subscribe(image_topic_right_,      10,  &PointCloudColorizer::imgRightCallback,     this);
    sub_img_left_   = nh_.subscribe(image_topic_left_,       10,  &PointCloudColorizer::imgLeftCallback,      this);
    sub_info_right_ = nh_.subscribe(camera_info_topic_right_, 1,  &PointCloudColorizer::camInfoRightCallback, this);
    sub_info_left_  = nh_.subscribe(camera_info_topic_left_,  1,  &PointCloudColorizer::camInfoLeftCallback,  this);
    sub_imu_        = nh_.subscribe(imu_topic_,             200,  &PointCloudColorizer::imuCallback,          this);

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);
    ROS_INFO("[ColoriseScan] Initialized. Publishing to %s", output_topic_.c_str());
}

// ── Odometry stubs (used by ColoriseMap, not by this node) ────────────────────
void PointCloudColorizer::odomCallback(const nav_msgs::OdometryConstPtr&) {}
bool PointCloudColorizer::interpolateOdometry(ros::Time, Eigen::Matrix4d&) { return false; }

// ── Camera info callbacks ─────────────────────────────────────────────────────

void PointCloudColorizer::camInfoRightCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    if (!cam_info_right_) {
        cam_info_right_ = *msg;
        ROS_INFO("[ColoriseScan] Right camera info (frame: %s, %dx%d)",
                 msg->header.frame_id.c_str(), msg->width, msg->height);
    }
}

void PointCloudColorizer::camInfoLeftCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    if (!cam_info_left_) {
        cam_info_left_ = *msg;
        ROS_INFO("[ColoriseScan] Left camera info (frame: %s, %dx%d)",
                 msg->header.frame_id.c_str(), msg->width, msg->height);
    }
}

// ── IMU buffer ────────────────────────────────────────────────────────────────

void PointCloudColorizer::imuCallback(const sensor_msgs::ImuConstPtr& msg) {
    imu_buffer_.push_back(msg);
    cleanOldMsgs(imu_buffer_, msg->header.stamp);
}

bool PointCloudColorizer::interpolateIMUOrientation(ros::Time t, Eigen::Quaterniond& q_out) {
    if (imu_buffer_.size() < 2) return false;

    if (t >= imu_buffer_.back()->header.stamp) {
        const auto& q = imu_buffer_.back()->orientation;
        q_out = Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized();
        return true;
    }

    for (size_t i = 0; i + 1 < imu_buffer_.size(); ++i) {
        if (imu_buffer_[i]->header.stamp <= t && imu_buffer_[i+1]->header.stamp >= t) {
            double dt_total = (imu_buffer_[i+1]->header.stamp - imu_buffer_[i]->header.stamp).toSec();
            double dt_query = (t - imu_buffer_[i]->header.stamp).toSec();
            double alpha    = (dt_total > 1e-9) ? dt_query / dt_total : 0.0;
            const auto& qb  = imu_buffer_[i]->orientation;
            const auto& qa  = imu_buffer_[i+1]->orientation;
            q_out = Eigen::Quaterniond(qb.w, qb.x, qb.y, qb.z).normalized()
                        .slerp(alpha,
                               Eigen::Quaterniond(qa.w, qa.x, qa.y, qa.z).normalized());
            return true;
        }
    }
    return false;
}

// ── IMU rotation-only motion compensation ────────────────────────────────────
//
// For the short camera-lidar timestamp offset (~10-100 ms), only the rotational
// component matters — translational displacement is typically < 5 mm, which is
// sub-pixel at any useful projection distance.
//
// Chain:
//   R_delta_world = R_world(t_cam)^T  *  R_world(t_lidar)
//   R_lidar_comp  = R_lidar_imu       *  R_delta_world  *  R_imu_lidar
//   T_out         = T_cam_lidar_static * T_lidar_comp   (translation = 0)
//
// Falls back to the static extrinsic if IMU data is unavailable.
// ─────────────────────────────────────────────────────────────────────────────
bool PointCloudColorizer::computeCamFromLidarIMU(ros::Time t_lidar, ros::Time t_cam,
                                                 const std::string& cam_frame,
                                                 Eigen::Matrix4d& T_out)
{
    // Static cam-from-lidar extrinsic
    Eigen::Matrix4d T_cam_lidar_static;
    if (calib_loaded_) {
        bool is_right = (cam_frame.find("ight") != std::string::npos);
        T_cam_lidar_static = is_right ? T_cam_lidar_right_ : T_cam_lidar_left_;
        ROS_INFO_ONCE("[ColoriseScan] [%s] extrinsic: hardcoded calibration.yaml",
                      cam_frame.c_str());
    } else {
        try {
            T_cam_lidar_static = tf2::transformToEigen(
                tf_buffer_.lookupTransform(cam_frame, lidar_frame_,
                                           ros::Time(0), ros::Duration(0.1))).matrix();
            ROS_INFO_ONCE("[ColoriseScan] [%s] extrinsic: TF '%s' -> '%s'",
                          cam_frame.c_str(), cam_frame.c_str(), lidar_frame_.c_str());
        } catch (const tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(2.0, "[ColoriseScan] TF cam<-lidar failed for %s: %s",
                              cam_frame.c_str(), ex.what());
            return false;
        }
    }

    const double dt = (t_lidar - t_cam).toSec();

    // No compensation needed for negligible offset
    if (std::fabs(dt) < 1e-4) {
        T_out = T_cam_lidar_static;
        return true;
    }

    // SLERP IMU orientations at both timestamps
    Eigen::Quaterniond q_at_lidar, q_at_cam;
    if (!interpolateIMUOrientation(t_lidar, q_at_lidar) ||
        !interpolateIMUOrientation(t_cam,   q_at_cam)) {
        ROS_WARN_THROTTLE(2.0, "[ColoriseScan] IMU orientation unavailable for dt=%.4f s "
                          "— using static extrinsic", dt);
        T_out = T_cam_lidar_static;
        return true;
    }

    // R_imu_from_lidar for frame conversion
    Eigen::Matrix3d R_imu_lidar;
    if (calib_loaded_) {
        R_imu_lidar = R_imu_lidar_;
    } else {
        try {
            R_imu_lidar = tf2::transformToEigen(
                tf_buffer_.lookupTransform(imu_frame_, lidar_frame_,
                                           ros::Time(0), ros::Duration(0.1))).matrix()
                          .block<3,3>(0,0);
            ROS_INFO_ONCE("[ColoriseScan] R_imu_lidar: TF '%s' -> '%s'",
                          imu_frame_.c_str(), lidar_frame_.c_str());
        } catch (const tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(2.0, "[ColoriseScan] TF imu<-lidar failed: %s — using identity",
                              ex.what());
            R_imu_lidar = Eigen::Matrix3d::Identity();
        }
    }

    // Rotation delta in world frame, then expressed in lidar frame
    Eigen::Matrix3d R_delta_world = q_at_cam.normalized().toRotationMatrix().transpose()
                                  * q_at_lidar.normalized().toRotationMatrix();
    Eigen::Matrix3d R_lidar_comp  = R_imu_lidar.transpose() * R_delta_world * R_imu_lidar;

    Eigen::Matrix4d T_lidar_comp = Eigen::Matrix4d::Identity();
    T_lidar_comp.block<3,3>(0,0) = R_lidar_comp;

    T_out = T_cam_lidar_static * T_lidar_comp;

    ROS_DEBUG_THROTTLE(1.0, "[ColoriseScan] IMU comp [%s]: rot=%.4f deg, dt=%.4f s",
        cam_frame.c_str(),
        Eigen::AngleAxisd(R_lidar_comp).angle() * 180.0 / M_PI, dt);

    return true;
}

// ── Cloud / image callbacks and sync ─────────────────────────────────────────

void PointCloudColorizer::cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    cloud_buffer_.push_back(msg);
    cleanOldMsgs(cloud_buffer_, msg->header.stamp);
    trySyncAndProcess();
}

void PointCloudColorizer::imgRightCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
    img_right_buffer_.push_back(msg);
    cleanOldMsgs(img_right_buffer_, msg->header.stamp);
}

void PointCloudColorizer::imgLeftCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
    img_left_buffer_.push_back(msg);
    cleanOldMsgs(img_left_buffer_, msg->header.stamp);
}

void PointCloudColorizer::trySyncAndProcess() {
    if (!cam_info_right_ || !cam_info_left_) {
        ROS_WARN_THROTTLE(5.0, "[ColoriseScan] Waiting for camera_info on both cameras...");
        return;
    }
    if (cloud_buffer_.empty() || img_right_buffer_.empty() || img_left_buffer_.empty()) return;

    for (auto it_cloud = cloud_buffer_.begin(); it_cloud != cloud_buffer_.end(); ++it_cloud) {
        ros::Time t_lidar = (*it_cloud)->header.stamp;

        auto img_right = findClosestBefore(img_right_buffer_, t_lidar);
        auto img_left  = findClosestBefore(img_left_buffer_,  t_lidar);
        if (!img_right || !img_left) continue;

        callback(*it_cloud, img_right, img_left, t_lidar,
                 img_right->header.stamp > img_left->header.stamp
                     ? img_right->header.stamp : img_left->header.stamp);
        cloud_buffer_.erase(cloud_buffer_.begin(), it_cloud + 1);
        return;
    }
}

template<typename T>
void PointCloudColorizer::cleanOldMsgs(std::deque<T>& buffer, ros::Time latest_time) {
    while (!buffer.empty() &&
           (latest_time - buffer.front()->header.stamp).toSec() > 2.0)
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

template<typename T>
typename T::value_type PointCloudColorizer::findClosestBefore(const T& buffer, ros::Time reference_time) {
    typename T::value_type best_match = nullptr;
    for (const auto& msg : buffer) {
        double diff = (reference_time - msg->header.stamp).toSec();
        if (diff >= 0.0 && diff <= max_time_offset_)
            if (!best_match || msg->header.stamp > best_match->header.stamp)
                best_match = msg;
    }
    return best_match;
}

// ── Main processing callback ──────────────────────────────────────────────────

void PointCloudColorizer::callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                                   const sensor_msgs::CompressedImageConstPtr& img_right_msg,
                                   const sensor_msgs::CompressedImageConstPtr& img_left_msg,
                                   ros::Time t_lidar,
                                   ros::Time t_cam)
{
    // Unpack point cloud
    pcl::PointCloud<PointXYZRGBIntensity>::Ptr in(new pcl::PointCloud<PointXYZRGBIntensity>);
    in->header.frame_id = cloud_msg->header.frame_id;
    in->is_dense = false; in->height = 1;

    const size_t point_step = cloud_msg->point_step;
    const size_t num_points = cloud_msg->width * cloud_msg->height;
    in->points.reserve(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        const uint8_t* ptr = &cloud_msg->data[i * point_step];
        PointXYZRGBIntensity pt;
        std::memcpy(&pt.x,            ptr +  0, sizeof(float));
        std::memcpy(&pt.y,            ptr +  4, sizeof(float));
        std::memcpy(&pt.z,            ptr +  8, sizeof(float));
        std::memcpy(&pt.intensity,    ptr + 16, sizeof(float));
        std::memcpy(&pt.t,            ptr + 20, sizeof(uint32_t));
        std::memcpy(&pt.reflectivity, ptr + 24, sizeof(uint16_t));
        std::memcpy(&pt.ring,         ptr + 26, sizeof(uint16_t));
        std::memcpy(&pt.ambient,      ptr + 28, sizeof(uint16_t));
        std::memcpy(&pt.range,        ptr + 32, sizeof(uint32_t));
        pt.rgb = 0;
        in->points.push_back(pt);
    }
    in->width = in->points.size();

    auto out = boost::make_shared<pcl::PointCloud<PointXYZRGBIntensity>>();
    out->header.frame_id = cloud_msg->header.frame_id;
    out->is_dense = false; out->height = 1;
    if (keep_uncolored_points_) out->points = in->points;

    std::vector<cv::Point3f> P3;
    P3.reserve(in->points.size());
    for (const auto& p : in->points)
        P3.emplace_back(p.x, p.y, p.z);

    // Decode images
    cv::Mat img_right = cv::imdecode(cv::Mat(img_right_msg->data), cv::IMREAD_COLOR);
    cv::Mat img_left  = cv::imdecode(cv::Mat(img_left_msg->data),  cv::IMREAD_COLOR);

    // Build intrinsics
    auto buildK = [](const sensor_msgs::CameraInfo& info) {
        cv::Mat_<double> K(3, 3);
        K << info.K[0], 0.0, info.K[2], 0.0, info.K[4], info.K[5], 0.0, 0.0, 1.0;
        return K;
    };
    auto buildDist = [](const sensor_msgs::CameraInfo& info) {
        cv::Mat d(info.D, true); return d.reshape(1, 1);
    };

    // Compute IMU-compensated cam-from-lidar transforms (one per camera, own timestamp)
    Eigen::Matrix4d T_right, T_left;
    if (!computeCamFromLidarIMU(t_lidar, img_right_msg->header.stamp,
                                cam_info_right_->header.frame_id, T_right) ||
        !computeCamFromLidarIMU(t_lidar, img_left_msg->header.stamp,
                                cam_info_left_->header.frame_id,  T_left)) {
        ROS_WARN_THROTTLE(2.0, "[ColoriseScan] Transform failed, skipping scan t=%.3f",
                          t_lidar.toSec());
        return;
    }

    // Colorize
    colorize(P3, img_right, T_right,
             buildK(*cam_info_right_), buildDist(*cam_info_right_),
             cam_info_right_->distortion_model,
             (int)cam_info_right_->width, (int)cam_info_right_->height,
             out, in, static_cast<float>(max_lidar_z_));
    colorize(P3, img_left, T_left,
             buildK(*cam_info_left_), buildDist(*cam_info_left_),
             cam_info_left_->distortion_model,
             (int)cam_info_left_->width, (int)cam_info_left_->height,
             out, in, static_cast<float>(max_lidar_z_));

    out->width = out->points.size();

    // Remove NaN and publish
    pcl::PointCloud<PointXYZRGBIntensity>::Ptr cleaned(new pcl::PointCloud<PointXYZRGBIntensity>);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*out, *cleaned, indices);

    sensor_msgs::PointCloud2 out_msg;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl::toPCLPointCloud2(*cleaned, pcl_pc2);
    pcl_conversions::fromPCL(pcl_pc2, out_msg);
    out_msg.header = cloud_msg->header;
    pub_.publish(out_msg);
}

// ── colorize ──────────────────────────────────────────────────────────────────

void PointCloudColorizer::colorize(const std::vector<cv::Point3f>& P3,
                                   const cv::Mat& img,
                                   const Eigen::Matrix4d& T_cam_from_lidar,
                                   const cv::Mat& K,
                                   const cv::Mat& dist,
                                   const std::string& distortion_model,
                                   int width, int height,
                                   pcl::PointCloud<PointXYZRGBIntensity>::Ptr& out,
                                   const pcl::PointCloud<PointXYZRGBIntensity>::Ptr& in,
                                   float source_z_max)
{
    Eigen::Matrix3d R_e = T_cam_from_lidar.block<3,3>(0,0);
    Eigen::Vector3d t_e = T_cam_from_lidar.block<3,1>(0,3);
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
        if (P3[i].z > source_z_max) continue;

        Eigen::Vector4d pt_cam = T_cam_from_lidar *
                                 Eigen::Vector4d(P3[i].x, P3[i].y, P3[i].z, 1.0);
        bool valid = pt_cam.z() > 0;
        int u = std::round(P2[i].x);
        int v = std::round(P2[i].y);
        valid &= (u >= 0 && u < width && v >= 0 && v < height);

        if (!keep_uncolored_points_ && !valid) continue;

        uint32_t rgb = 0;
        if (valid) {
            cv::Vec3b c = img.at<cv::Vec3b>(v, u);
            rgb = (uint32_t(c[2]) << 16 | uint32_t(c[1]) << 8 | uint32_t(c[0]));
        }

        if (keep_uncolored_points_) {
            if (rgb != 0) out->points[i].rgb = rgb;
        } else if (valid) {
            PointXYZRGBIntensity pt = in->points[i];
            pt.rgb = rgb;
            out->points.push_back(pt);
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    ros::init(argc, argv, "colorize_pointcloud_node");
    ros::NodeHandle nh("~");

    std::string pkg = ros::package::getPath("point_cloud_projection");
    std::string config_path;
    nh.param("config_path", config_path, pkg + "/configs/config.yaml");

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    double startup_delay = 0.0;
    if (fs.isOpened()) fs["initial_startup_delay"] >> startup_delay;
    fs.release();

    ROS_INFO("[ColoriseScan] Waiting %.2f sec for sensor startup...", startup_delay);
    ros::Duration(startup_delay).sleep();

    PointCloudColorizer node(nh);
    ros::spin();
    return 0;
}
