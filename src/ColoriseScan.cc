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
    fs["odom_topic"]            >> odom_topic_;
    fs["use_odom_compensation"] >> use_odom_compensation_;

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

    sub_cloud_      = nh_.subscribe(cloud_topic_,              10,  &PointCloudColorizer::cloudCallback,        this);
    sub_img_right_  = nh_.subscribe(image_topic_right_,        10,  &PointCloudColorizer::imgRightCallback,     this);
    sub_img_left_   = nh_.subscribe(image_topic_left_,         10,  &PointCloudColorizer::imgLeftCallback,      this);
    sub_info_right_ = nh_.subscribe(camera_info_topic_right_,   1,  &PointCloudColorizer::camInfoRightCallback, this);
    sub_info_left_  = nh_.subscribe(camera_info_topic_left_,    1,  &PointCloudColorizer::camInfoLeftCallback,  this);
    sub_imu_        = nh_.subscribe(imu_topic_,               200,  &PointCloudColorizer::imuCallback,          this);
    sub_odom_ = nh_.subscribe(odom_topic_, 200, &PointCloudColorizer::odomCallback, this);

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);
    ROS_INFO("Initialized and publishing to %s", output_topic_.c_str());
}

void PointCloudColorizer::odomCallback(const nav_msgs::OdometryConstPtr& msg) {
    odom_buffer_.push_back(msg);
    cleanOldMsgs(odom_buffer_, msg->header.stamp);
}

// Interpolates odometry pose at time t using linear+SLERP blending
bool PointCloudColorizer::interpolateOdometry(ros::Time t, Eigen::Matrix4d& T_out) {
    if (odom_buffer_.size() < 2) return false;

    // Use latest if t is beyond buffer
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

            double dt_total = (odom_buffer_[i+1]->header.stamp -
                               odom_buffer_[i]->header.stamp).toSec();
            double dt_query = (t - odom_buffer_[i]->header.stamp).toSec();
            double alpha    = (dt_total > 1e-9) ? dt_query / dt_total : 0.0;

            const auto& p0 = odom_buffer_[i]->pose.pose;
            const auto& p1 = odom_buffer_[i+1]->pose.pose;

            // Interpolate translation linearly
            Eigen::Vector3d t0(p0.position.x, p0.position.y, p0.position.z);
            Eigen::Vector3d t1(p1.position.x, p1.position.y, p1.position.z);
            Eigen::Vector3d t_interp = t0 + alpha * (t1 - t0);

            // Interpolate rotation via SLERP
            Eigen::Quaterniond q0(p0.orientation.w, p0.orientation.x,
                                  p0.orientation.y, p0.orientation.z);
            Eigen::Quaterniond q1(p1.orientation.w, p1.orientation.x,
                                  p1.orientation.y, p1.orientation.z);
            Eigen::Quaterniond q_interp = q0.normalized().slerp(alpha, q1.normalized());

            T_out = Eigen::Matrix4d::Identity();
            T_out.block<3,3>(0,0) = q_interp.toRotationMatrix();
            T_out.block<3,1>(0,3) = t_interp;
            return true;
        }
    }
    return false;
}

// Compute compensation from odometry:
// T_compensation = T_odom_lidar_at_cam * T_odom_lidar_at_lidar^-1
// i.e. undo the motion that happened between t_cam and t_lidar,
// expressed in the lidar frame.
Eigen::Matrix4d PointCloudColorizer::computeMotionCompensationOdom(ros::Time t_lidar, ros::Time t_cam) {
    Eigen::Matrix4d T_world_base_at_lidar, T_world_base_at_cam;

    if (!interpolateOdometry(t_lidar, T_world_base_at_lidar) ||
        !interpolateOdometry(t_cam,   T_world_base_at_cam)) {
        ROS_WARN_THROTTLE(2.0, "Odometry unavailable for motion compensation, using identity");
        return Eigen::Matrix4d::Identity();
    }

    // Get static transform from base_link to lidar frame
    Eigen::Matrix4d T_base_lidar = Eigen::Matrix4d::Identity();
    try {
        geometry_msgs::TransformStamped ts =
            tf_buffer_.lookupTransform(
                odom_buffer_.front()->child_frame_id, // base_link (or whatever odom reports)
                lidar_frame_,
                ros::Time(0), ros::Duration(0.1));
        T_base_lidar = tf2::transformToEigen(ts).matrix();
    } catch (const tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(2.0, "TF base<-lidar failed: %s. Using identity.", ex.what());
        return Eigen::Matrix4d::Identity();
    }

    // World pose of lidar at each time:
    // T_world_lidar = T_world_base * T_base_lidar
    Eigen::Matrix4d T_world_lidar_at_lidar = T_world_base_at_lidar * T_base_lidar;
    Eigen::Matrix4d T_world_lidar_at_cam   = T_world_base_at_cam   * T_base_lidar;

    // T_compensation: transform points from lidar-time pose back to cam-time pose
    // p_at_cam = T_world_lidar_at_cam^-1 * T_world_lidar_at_lidar * p_at_lidar
    Eigen::Matrix4d T_compensation = T_world_lidar_at_cam.inverse() * T_world_lidar_at_lidar;

    ROS_DEBUG_THROTTLE(1.0, "Odom compensation: |t|=%.4f m, rot=%.4f deg, dt=%.4f s",
                       T_compensation.block<3,1>(0,3).norm(),
                       Eigen::AngleAxisd(T_compensation.block<3,3>(0,0)).angle() * 180.0 / M_PI,
                       (t_lidar - t_cam).toSec());

    return T_compensation;
}

void PointCloudColorizer::camInfoRightCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    if (!cam_info_right_) {
        cam_info_right_ = *msg;
        ROS_INFO("Right camera info (frame: %s, %dx%d)", msg->header.frame_id.c_str(), msg->width, msg->height);
    }
}

void PointCloudColorizer::camInfoLeftCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    if (!cam_info_left_) {
        cam_info_left_ = *msg;
        ROS_INFO("Left camera info (frame: %s, %dx%d)", msg->header.frame_id.c_str(), msg->width, msg->height);
    }
}

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
            Eigen::Quaterniond q0(qb.w, qb.x, qb.y, qb.z);
            Eigen::Quaterniond q1(qa.w, qa.x, qa.y, qa.z);
            q_out = q0.normalized().slerp(alpha, q1.normalized());
            return true;
        }
    }
    return false;
}

// Full 6-DOF motion compensation.
// Rotation:    AHRS SLERP between t_cam and t_lidar.
// Translation: Trapezoid integration of gravity-removed IMU accel over [t_cam, t_lidar].
//              Assumes zero velocity at t_cam (valid for slow agricultural robot over ~100 ms).
Eigen::Matrix4d PointCloudColorizer::computeMotionCompensation(ros::Time t_lidar, ros::Time t_cam) {
    static const Eigen::Vector3d GRAVITY(0.0, 0.0, 9.81); // world Z-up, m/s^2

    // Collect IMU samples in the [t_cam, t_lidar] window
    std::vector<const sensor_msgs::Imu*> window;
    for (const auto& msg : imu_buffer_)
        if (msg->header.stamp >= t_cam && msg->header.stamp <= t_lidar)
            window.push_back(msg.get());

    // AHRS orientations at both endpoints
    Eigen::Quaterniond q_at_lidar, q_at_cam;
    if (!interpolateIMUOrientation(t_lidar, q_at_lidar) ||
        !interpolateIMUOrientation(t_cam,   q_at_cam)) {
        ROS_WARN_THROTTLE(2.0, "IMU orientation unavailable, using identity compensation");
        return Eigen::Matrix4d::Identity();
    }

    // Delta rotation in world frame: undo motion between t_cam and t_lidar
    Eigen::Matrix3d R_delta_world = q_at_cam.toRotationMatrix().transpose() *
                                    q_at_lidar.toRotationMatrix();

    // Get static T_imu_lidar from TF
    Eigen::Matrix4d T_imu_lidar = Eigen::Matrix4d::Identity();
    try {
        geometry_msgs::TransformStamped ts =
            tf_buffer_.lookupTransform(imu_frame_, lidar_frame_, ros::Time(0), ros::Duration(0.1));
        T_imu_lidar = tf2::transformToEigen(ts).matrix();
    } catch (const tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(2.0, "TF imu<-lidar failed: %s. Using identity.", ex.what());
        return Eigen::Matrix4d::Identity();
    }

    Eigen::Matrix3d R_imu_lidar = T_imu_lidar.block<3,3>(0,0);

    // Express rotation delta in lidar frame
    Eigen::Matrix3d R_delta_lidar = R_imu_lidar.transpose() * R_delta_world * R_imu_lidar;

    // Double-integrate accel over window to get displacement in world frame
    Eigen::Vector3d translation_world = Eigen::Vector3d::Zero();

    if (window.size() >= 2) {
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero(); // zero initial velocity at t_cam

        for (size_t i = 0; i + 1 < window.size(); ++i) {
            double dt = (window[i+1]->header.stamp - window[i]->header.stamp).toSec();
            if (dt <= 0.0 || dt > 0.1) continue;

            Eigen::Quaterniond q_i, q_j;
            interpolateIMUOrientation(window[i]->header.stamp,   q_i);
            interpolateIMUOrientation(window[i+1]->header.stamp, q_j);

            const auto& ai = window[i]->linear_acceleration;
            const auto& aj = window[i+1]->linear_acceleration;

            // Raw accel in IMU frame -> world frame -> remove gravity
            Eigen::Vector3d a_i = q_i.toRotationMatrix() * Eigen::Vector3d(ai.x, ai.y, ai.z) - GRAVITY;
            Eigen::Vector3d a_j = q_j.toRotationMatrix() * Eigen::Vector3d(aj.x, aj.y, aj.z) - GRAVITY;
            Eigen::Vector3d a_avg = 0.5 * (a_i + a_j);

            // Trapezoid integration
            translation_world += velocity * dt + 0.5 * a_avg * dt * dt;
            velocity          += a_avg * dt;
        }
    }

    // Bring displacement into lidar frame
    Eigen::Vector3d translation_lidar =
        R_imu_lidar.transpose() *
        q_at_cam.toRotationMatrix().transpose() *
        translation_world;

    // Assemble: p_at_cam = R_delta * p_at_lidar - translation
    Eigen::Matrix4d T_compensation = Eigen::Matrix4d::Identity();
    T_compensation.block<3,3>(0,0) = R_delta_lidar;
    T_compensation.block<3,1>(0,3) = -translation_lidar;

    ROS_DEBUG_THROTTLE(1.0, "IMU compensation: |t|=%.4f m, rot=%.4f deg, dt=%.4f s",
                       translation_lidar.norm(),
                       Eigen::AngleAxisd(R_delta_lidar).angle() * 180.0 / M_PI,
                       (t_lidar - t_cam).toSec());

    return T_compensation;
}

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
        ROS_WARN_THROTTLE(5.0, "Waiting for camera_info on both cameras...");
        return;
    }
    if (cloud_buffer_.empty() || img_right_buffer_.empty() || img_left_buffer_.empty()) return;

    for (auto it_cloud = cloud_buffer_.begin(); it_cloud != cloud_buffer_.end(); ++it_cloud) {
        ros::Time t_lidar = (*it_cloud)->header.stamp;

        auto img_right = findClosestBefore(img_right_buffer_, t_lidar);
        auto img_left  = findClosestBefore(img_left_buffer_,  t_lidar);
        if (!img_right || !img_left) continue;

        ros::Time t_cam = (img_right->header.stamp > img_left->header.stamp)
                          ? img_right->header.stamp : img_left->header.stamp;

        Eigen::Matrix4d T_compensation = use_odom_compensation_
            ? computeMotionCompensationOdom(t_lidar, t_cam)
            : computeMotionCompensation(t_lidar, t_cam);
        callback(*it_cloud, img_right, img_left, T_compensation);
        cloud_buffer_.erase(cloud_buffer_.begin(), it_cloud + 1);
        return;
    }
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

void PointCloudColorizer::callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                                   const sensor_msgs::CompressedImageConstPtr& img_right_msg,
                                   const sensor_msgs::CompressedImageConstPtr& img_left_msg,
                                   const Eigen::Matrix4d& T_compensation)
{
    pcl::PointCloud<PointXYZRGBIntensity>::Ptr in(new pcl::PointCloud<PointXYZRGBIntensity>);
    in->header.frame_id = cloud_msg->header.frame_id;
    in->is_dense = false; in->height = 1;

    const size_t point_step = cloud_msg->point_step;
    const size_t num_points = cloud_msg->width * cloud_msg->height;
    const auto& data = cloud_msg->data;
    in->points.reserve(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        const uint8_t* ptr = &data[i * point_step];
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
    else out->points.clear();

    // Apply 6-DOF motion compensation to bring points to t_cam pose
    std::vector<cv::Point3f> P3;
    P3.reserve(in->points.size());
    for (const auto& p : in->points) {
        Eigen::Vector4d pc = T_compensation * Eigen::Vector4d(p.x, p.y, p.z, 1.0);
        P3.emplace_back(static_cast<float>(pc.x()), static_cast<float>(pc.y()), static_cast<float>(pc.z()));
    }

    cv::Mat img_right = cv::imdecode(cv::Mat(img_right_msg->data), cv::IMREAD_COLOR);
    cv::Mat img_left  = cv::imdecode(cv::Mat(img_left_msg->data),  cv::IMREAD_COLOR);

    auto buildK = [](const sensor_msgs::CameraInfo& info) {
        cv::Mat_<double> K(3, 3);
        K << info.K[0], 0.0, info.K[2], 0.0, info.K[4], info.K[5], 0.0, 0.0, 1.0;
        return K;
    };
    auto buildDist = [](const sensor_msgs::CameraInfo& info) {
        cv::Mat d(info.D, true); return d.reshape(1, 1);
    };
    auto lookupT = [&](const std::string& camera_frame) -> Eigen::Matrix4d {
        try {
            geometry_msgs::TransformStamped ts =
                tf_buffer_.lookupTransform(lidar_frame_, camera_frame, ros::Time(0), ros::Duration(0.1));
            return tf2::transformToEigen(ts).matrix();
        } catch (const tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(2.0, "TF lookup failed (%s -> %s): %s", lidar_frame_.c_str(), camera_frame.c_str(), ex.what());
            return Eigen::Matrix4d::Identity();
        }
    };

    cv::Mat K_right = buildK(*cam_info_right_), K_left = buildK(*cam_info_left_);
    cv::Mat dist_right = buildDist(*cam_info_right_), dist_left = buildDist(*cam_info_left_);
    Eigen::Matrix4d T_right = lookupT(cam_info_right_->header.frame_id);
    Eigen::Matrix4d T_left  = lookupT(cam_info_left_->header.frame_id);

    colorize(P3, img_right, T_right, K_right, dist_right, cam_info_right_->distortion_model,
             (int)cam_info_right_->width, (int)cam_info_right_->height, true,  false, out, in);
    colorize(P3, img_left,  T_left,  K_left,  dist_left,  cam_info_left_->distortion_model,
             (int)cam_info_left_->width,  (int)cam_info_left_->height,  false, true,  out, in);

    out->width = out->points.size();

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

void PointCloudColorizer::colorize(const std::vector<cv::Point3f>& P3, const cv::Mat& img,
                                   const Eigen::Matrix4d& T_camera_lidar, const cv::Mat& K, const cv::Mat& dist,
                                   const std::string& distortion_model, int width, int height,
                                   bool is_right, bool mirror_u,
                                   pcl::PointCloud<PointXYZRGBIntensity>::Ptr& out,
                                   const pcl::PointCloud<PointXYZRGBIntensity>::Ptr& in)
{
    Eigen::Matrix4d T = T_camera_lidar.inverse();
    Eigen::Matrix3d R_e = T.block<3,3>(0,0);
    Eigen::Vector3d t_e = T.block<3,1>(0,3);
    cv::Mat R_cv, rvec, tvec(3,1,CV_64F);
    cv::eigen2cv(R_e, R_cv); cv::Rodrigues(R_cv, rvec);
    for (int i = 0; i < 3; ++i) tvec.at<double>(i,0) = t_e(i);

    std::vector<cv::Point2f> P2;
    if (distortion_model == "equidistant")
        cv::fisheye::projectPoints(P3, P2, rvec, tvec, K, dist);
    else
        cv::projectPoints(P3, P2, rvec, tvec, K, dist);

    for (size_t i = 0; i < P2.size(); ++i) {
        if (P3[i].z > max_lidar_z_) continue;
        Eigen::Vector4d pt_cam = T * Eigen::Vector4d(P3[i].x, P3[i].y, P3[i].z, 1.0);
        bool valid = pt_cam.z() > 0;
        int u = std::round(P2[i].x), v = std::round(P2[i].y);
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
            pt.rgb = rgb; out->points.push_back(pt);
        }
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "colorize_pointcloud_node"); // or colorise_map_node
    ros::NodeHandle nh("~");
    // Build default config path relative to package
    std::string package_path = ros::package::getPath("point_cloud_projection");
    std::string default_config = package_path + "/configs/config.yaml";
    std::string config_path;
    nh.param("config_path", config_path, default_config);
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    double startup_delay = 0.0;
    if (fs.isOpened()) fs["initial_startup_delay"] >> startup_delay;
    fs.release();
    ROS_INFO("Waiting %.2f sec for initial buffer fill...", startup_delay);
    ros::Duration(startup_delay).sleep();
    PointCloudColorizer node(nh);
    ros::spin();
    return 0;
}