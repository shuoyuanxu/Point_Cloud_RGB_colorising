// colorise_map_node.cpp
#include "colorise.h"
#include <signal.h>
#include "utils.h"

static PointCloudColorizer* global_instance = nullptr;

void signalHandler(int sig) {
    if (global_instance) global_instance->saveFinalMap();
    ros::shutdown();
}

PointCloudColorizer::PointCloudColorizer(ros::NodeHandle& nh) : nh_(nh) {
    nh_.param("config_path", config_path_, std::string("../configs/config.yaml"));

    cv::FileStorage fs(config_path_, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        ROS_ERROR("Unable to open config file: %s", config_path_.c_str());
        ros::shutdown(); return;
    }

    fs["max_time_offset"] >> max_time_offset_;
    fs["initial_startup_delay"] >> initial_startup_delay_;
    fs["image_topic_right"] >> image_topic_right_;
    fs["image_topic_left"] >> image_topic_left_;
    fs["output_topic"] >> output_topic_;
    fs["odom_topic"] >> odom_topic_;
    fs["min_color_frames"] >> min_color_frames_;
    fs["save_pcd_path"] >> save_path_;
    fs["map_pcd_path"] >> map_pcd_path_;

    map_points_.reset(new pcl::PointCloud<PointXYZRGBIntensity>());
    if (pcl::io::loadPCDFile<PointXYZRGBIntensity>(map_pcd_path_, *map_points_) == -1) {
        ROS_ERROR_STREAM("Failed to load map PCD file from " << map_pcd_path_);
        ros::shutdown(); return;
    }
    ROS_INFO_STREAM("Loaded map with " << map_points_->size() << " points from " << map_pcd_path_);

    fs["intrinsics_right"] >> intr_right_;
    cv_K_right_ = (cv::Mat_<double>(3,3) << intr_right_[0],0,intr_right_[2],0,intr_right_[1],intr_right_[3],0,0,1);
    fs["distortion_coeffs_right"] >> dist_right_; distCoeffs_right_ = cv::Mat(dist_right_);
    fs["distortion_model_right"] >> distortion_model_right_;
    fs["resolution_right"] >> resolution_right_;
    width_right_ = resolution_right_[0]; height_right_ = resolution_right_[1];
    parseMat(fs, "T_lidar_camera_right", T_lidar_camera_right_);

    fs["intrinsics_left"] >> intr_left_;
    cv_K_left_ = (cv::Mat_<double>(3,3) << intr_left_[0],0,intr_left_[2],0,intr_left_[1],intr_left_[3],0,0,1);
    fs["distortion_coeffs_left"] >> dist_left_; distCoeffs_left_ = cv::Mat(dist_left_);
    fs["distortion_model_left"] >> distortion_model_left_;
    fs["resolution_left"] >> resolution_left_;
    width_left_ = resolution_left_[0]; height_left_ = resolution_left_[1];
    parseMat(fs, "T_lidar_camera_left", T_lidar_camera_left_);

    path_sub_ = nh_.subscribe(odom_topic_, 10, &PointCloudColorizer::pathCallback, this);
    sub_img_right_ = nh_.subscribe(image_topic_right_, 10, &PointCloudColorizer::imgRightCallback, this);
    sub_img_left_ = nh_.subscribe(image_topic_left_, 10, &PointCloudColorizer::imgLeftCallback, this);
    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);

    accumulated_map_.reset(new pcl::PointCloud<PointXYZRGBIntensity>());
    global_instance = this;
    signal(SIGINT, signalHandler);

    color_timer_ = nh_.createTimer(ros::Duration(0.2), &PointCloudColorizer::colorizeFromPCD, this);
    ROS_INFO("ColoriseMap running from pre-mapped PCD.");
    fs.release();
}

void PointCloudColorizer::pathCallback(const nav_msgs::PathConstPtr& msg) {
    latest_path_ = *msg;
}

void PointCloudColorizer::imgRightCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
    img_right_buffer_.push_back(msg);
    cleanOldMsgs(img_right_buffer_, msg->header.stamp);
}

void PointCloudColorizer::imgLeftCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
    img_left_buffer_.push_back(msg);
    cleanOldMsgs(img_left_buffer_, msg->header.stamp);
}

void PointCloudColorizer::saveFinalMap() {
    accumulated_map_->width = accumulated_map_->points.size();
    accumulated_map_->height = 1;
    accumulated_map_->is_dense = false;
    pcl::io::savePCDFileBinary(save_path_, *accumulated_map_);
    ROS_INFO("Saved final colored map to %s", save_path_.c_str());
}

bool PointCloudColorizer::lookupPose(const ros::Time& t, Eigen::Matrix4d& T_out) {
    if (latest_path_.poses.empty()) return false;
    double min_diff = 1e9;
    geometry_msgs::PoseStamped best;
    for (const auto& p : latest_path_.poses) {
        double dt = fabs((p.header.stamp - t).toSec());
        if (dt < min_diff) {
            min_diff = dt;
            best = p;
        }
    }
    if (min_diff > max_time_offset_) return false;
    const auto& q = best.pose.orientation;
    const auto& tr = best.pose.position;
    Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
    Eigen::Matrix3d R = quat.toRotationMatrix();
    T_out.setIdentity();
    T_out.block<3,3>(0,0) = R;
    T_out.block<3,1>(0,3) = Eigen::Vector3d(tr.x, tr.y, tr.z);
    return true;
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

void PointCloudColorizer::colorize(const std::vector<cv::Point3f>& P3, const cv::Mat& img,
                                   const Eigen::Matrix4d& T_camera_lidar, const cv::Mat& K, const cv::Mat& dist,
                                   const std::string& distortion_model, int width, int height,
                                   bool is_right, bool mirror_u,
                                   pcl::PointCloud<PointXYZRGBIntensity>::Ptr& out,
                                   const pcl::PointCloud<PointXYZRGBIntensity>::Ptr& in) {
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
        Eigen::Vector4d pt_lidar(P3[i].x, P3[i].y, P3[i].z, 1.0);
        Eigen::Vector4d pt_cam = T * pt_lidar;
        if (pt_cam.z() <= 0) continue;

        int u = static_cast<int>(std::round(P2[i].x));
        int v = static_cast<int>(std::round(P2[i].y));
        if (u < 0 || u >= width || v < 0 || v >= height) continue;
        cv::Vec3b c = img.at<cv::Vec3b>(v, u);
        PointXYZRGBIntensity pt;
        pt.x = P3[i].x; pt.y = P3[i].y; pt.z = P3[i].z;
        pt.rgb = (static_cast<uint32_t>(c[2]) << 16 |
                  static_cast<uint32_t>(c[1]) << 8  |
                  static_cast<uint32_t>(c[0]));
        pt.intensity = in->points[i].intensity;
        out->points.push_back(pt);
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

void PointCloudColorizer::colorizeFromPCD(const ros::TimerEvent&) {
    if (!map_points_ || img_right_buffer_.empty() || img_left_buffer_.empty()) return;

    ros::Time t = img_right_buffer_.front()->header.stamp;
    Eigen::Matrix4d pose;
    if (!lookupPose(t, pose)) {
        ROS_WARN_THROTTLE(2.0, "No valid pose for current image time");
        return;
    }

    cv::Mat img_right = cv::imdecode(cv::Mat(img_right_buffer_.front()->data), cv::IMREAD_COLOR);
    cv::Mat img_left = cv::imdecode(cv::Mat(img_left_buffer_.front()->data), cv::IMREAD_COLOR);

    std::vector<cv::Point3f> P3;
    for (const auto& pt : map_points_->points)
        P3.emplace_back(pt.x, pt.y, pt.z);

    pcl::PointCloud<PointXYZRGBIntensity>::Ptr temp_colored(new pcl::PointCloud<PointXYZRGBIntensity>());
    colorize(P3, img_right, T_lidar_camera_right_, cv_K_right_, distCoeffs_right_, distortion_model_right_,
             width_right_, height_right_, true, false, temp_colored, map_points_);

    colorize(P3, img_left, T_lidar_camera_left_, cv_K_left_, distCoeffs_left_, distortion_model_left_,
             width_left_, height_left_, false, true, temp_colored, map_points_);

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

    ROS_INFO_STREAM_THROTTLE(2.0, "[DEBUG] Accumulated map now has " << accumulated_map_->points.size() << " points");

    sensor_msgs::PointCloud2 out_msg;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl::toPCLPointCloud2(*accumulated_map_, pcl_pc2);
    pcl_conversions::fromPCL(pcl_pc2, out_msg);
    out_msg.header.stamp = t;
    out_msg.header.frame_id = "map";
    pub_.publish(out_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "colorise_map_node");
    ros::NodeHandle nh("~");

    std::string config_path;
    nh.param("config_path", config_path, std::string("../configs/config.yaml"));

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    double startup_delay = 0.1;
    if (fs.isOpened()) fs["initial_startup_delay"] >> startup_delay;
    fs.release();

    ROS_INFO("Waiting %.2f sec for initial buffer fill...", startup_delay);
    ros::Duration(startup_delay).sleep();

    PointCloudColorizer node(nh);
    ros::spin();
    return 0;
}