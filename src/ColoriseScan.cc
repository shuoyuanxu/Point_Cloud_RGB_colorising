#include "colorise.h"
#include "utils.h"

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
    fs["pointcloud_topic"] >> cloud_topic_;
    fs["output_topic"] >> output_topic_;
    fs["keep_uncolored_points"] >> keep_uncolored_points_; 

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
    fs.release();

    sub_cloud_ = nh_.subscribe(cloud_topic_, 10, &PointCloudColorizer::cloudCallback, this);
    sub_img_right_ = nh_.subscribe(image_topic_right_, 10, &PointCloudColorizer::imgRightCallback, this);
    sub_img_left_ = nh_.subscribe(image_topic_left_, 10, &PointCloudColorizer::imgLeftCallback, this);

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);
    ROS_INFO("Initialized and publishing to %s", output_topic_.c_str());
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
    if (cloud_buffer_.empty() || img_right_buffer_.empty() || img_left_buffer_.empty()) return;
    for (auto it_cloud = cloud_buffer_.begin(); it_cloud != cloud_buffer_.end(); ++it_cloud) {
        ros::Time t = (*it_cloud)->header.stamp;
        auto img_right = findClosest(img_right_buffer_, t);
        auto img_left = findClosest(img_left_buffer_, t);
        if (!img_right || !img_left) continue;
        callback(*it_cloud, img_right, img_left);
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

void PointCloudColorizer::callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                                   const sensor_msgs::CompressedImageConstPtr& img_right_msg,
                                   const sensor_msgs::CompressedImageConstPtr& img_left_msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr in_raw(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*cloud_msg, *in_raw);

    pcl::PointCloud<PointXYZRGBIntensity>::Ptr in(new pcl::PointCloud<PointXYZRGBIntensity>);
    in->points.reserve(in_raw->size());
    for (const auto& p : in_raw->points) {
        PointXYZRGBIntensity pt;
        pt.x = p.x;
        pt.y = p.y;
        pt.z = p.z;
        pt.intensity = p.intensity;
        pt.rgb = 0.0f;
        in->points.push_back(pt);
    }

    auto out = boost::make_shared<pcl::PointCloud<PointXYZRGBIntensity>>();
    out->header.frame_id = cloud_msg->header.frame_id;
    out->is_dense = false;
    out->height = 1;

    if (keep_uncolored_points_) {
        out->points = in->points; // pre-fill with all points
    } else {
        out->points.clear(); // selectively populate
    }

    std::vector<cv::Point3f> P3;
    for (auto& p : in->points) P3.emplace_back(p.x, p.y, p.z);

    cv::Mat img_right = cv::imdecode(cv::Mat(img_right_msg->data), cv::IMREAD_COLOR);
    cv::Mat img_left  = cv::imdecode(cv::Mat(img_left_msg->data),  cv::IMREAD_COLOR);
    
    colorize(P3, img_right, T_lidar_camera_right_, cv_K_right_, distCoeffs_right_, distortion_model_right_,
                width_right_, height_right_, true, false, out, in);
    colorize(P3, img_left, T_lidar_camera_left_, cv_K_left_, distCoeffs_left_, distortion_model_left_,
                width_left_, height_left_, false, true, out, in);

    out->width = out->points.size();

    sensor_msgs::PointCloud2 out_msg;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl::toPCLPointCloud2(*out, pcl_pc2);
    pcl_conversions::fromPCL(pcl_pc2, out_msg);
    out_msg.header = cloud_msg->header;
    pub_.publish(out_msg);
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
        Eigen::Vector4d pt_cam = T * Eigen::Vector4d(P3[i].x, P3[i].y, P3[i].z, 1.0);
        bool valid = pt_cam.z() > 0;
        int u = std::round(P2[i].x), v = std::round(P2[i].y);
        valid &= (u >= 0 && u < width && v >= 0 && v < height);

        if (!keep_uncolored_points_ && !valid) continue; // Skip uncolored points if flag is false

        uint32_t rgb = 0; // default black
        if (valid) {
            cv::Vec3b c = img.at<cv::Vec3b>(v, u);
            rgb = (uint32_t(c[2]) << 16 | uint32_t(c[1]) << 8 | uint32_t(c[0]));
        }

        if (keep_uncolored_points_) {
            if (rgb != 0) // only overwrite if color is valid
                out->points[i].rgb = rgb;
        } else if (valid) {
            PointXYZRGBIntensity pt = in->points[i];
            pt.rgb = rgb;
            out->points.push_back(pt);
        }
    }
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "colorize_pointcloud_node");
    ros::NodeHandle nh("~");

    std::string config_path;
    nh.param("config_path", config_path, std::string("../configs/config.yaml"));
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    double startup_delay;
    if (fs.isOpened()) fs["initial_startup_delay"] >> startup_delay;
    fs.release();

    ROS_INFO("Waiting %.2f sec for initial buffer fill...", startup_delay);
    ros::Duration(startup_delay).sleep();

    PointCloudColorizer node(nh);
    ros::spin();
    return 0;
}
