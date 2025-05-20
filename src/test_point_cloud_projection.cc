#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CompressedImage.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// Parse a 4x4 matrix from YAML (nested 4x4 or flat sequence)
void parseMat(const cv::FileStorage& fs, const std::string& name, Eigen::Matrix4d& M) {
    cv::FileNode node = fs[name];
    if (node.empty()) throw std::runtime_error("Missing matrix " + name);
    std::vector<double> v;
    if (node.isSeq() && node[0].isSeq()) {
        for (auto row : node) for (auto val : row) v.push_back((double)val);
    } else {
        node >> v;
    }
    if (v.size() != 16) {
        std::ostringstream oss;
        oss << name << " must have 16 elements, but got " << v.size();
        throw std::runtime_error(oss.str());
    }
    M = Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>>(v.data());
}

class PointCloudColorizer {
public:
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::PointCloud2,
        sensor_msgs::CompressedImage,
        sensor_msgs::CompressedImage
    > SyncPolicy;

    PointCloudColorizer(ros::NodeHandle& nh) : nh_(nh) {
        nh_.param("config_path", config_path_, std::string("../configs/config.yaml"));
        cv::FileStorage fs(config_path_, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            ROS_ERROR("Unable to open %s", config_path_.c_str());
            ros::shutdown(); return;
        }

        fs["image_topic_right"] >> image_topic_right_;
        fs["image_topic_left"] >> image_topic_left_;
        fs["pointcloud_topic"] >> cloud_topic_;
        fs["output_topic"] >> output_topic_;

        fs["intrinsics_right"] >> intr_right_; cv_K_right_ = (cv::Mat_<double>(3,3) << intr_right_[0],0,intr_right_[2],0,intr_right_[1],intr_right_[3],0,0,1);
        fs["distortion_coeffs_right"] >> dist_right_; distCoeffs_right_ = cv::Mat(dist_right_);
        fs["distortion_model_right"] >> distortion_model_right_;
        fs["resolution_right"] >> resolution_right_;
        width_right_ = resolution_right_[0]; height_right_ = resolution_right_[1];
        parseMat(fs, "T_lidar_camera_right", T_lidar_camera_right_);

        fs["intrinsics_left"] >> intr_left_; cv_K_left_ = (cv::Mat_<double>(3,3) << intr_left_[0],0,intr_left_[2],0,intr_left_[1],intr_left_[3],0,0,1);
        fs["distortion_coeffs_left"] >> dist_left_; distCoeffs_left_ = cv::Mat(dist_left_);
        fs["distortion_model_left"] >> distortion_model_left_;
        fs["resolution_left"] >> resolution_left_;
        width_left_ = resolution_left_[0]; height_left_ = resolution_left_[1];
        parseMat(fs, "T_lidar_camera_left", T_lidar_camera_left_);
        fs.release();

        sub_cloud_.subscribe(nh_, cloud_topic_, 1);
        sub_img_right_.subscribe(nh_, image_topic_right_, 1);
        sub_img_left_.subscribe(nh_, image_topic_left_, 1);
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), sub_cloud_, sub_img_right_, sub_img_left_));
        sync_->registerCallback(boost::bind(&PointCloudColorizer::callback, this, _1, _2, _3));

        pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);
        ROS_INFO("Publishing colorized cloud on %s", output_topic_.c_str());
    }

private:
    void callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                    const sensor_msgs::CompressedImageConstPtr& img_right_msg,
                    const sensor_msgs::CompressedImageConstPtr& img_left_msg) {

        pcl::PointCloud<pcl::PointXYZ>::Ptr in(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *in);
        auto out = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        out->header.frame_id = cloud_msg->header.frame_id;
        out->is_dense = false;
        out->height = 1;

        cv::Mat img_right = cv::imdecode(cv::Mat(img_right_msg->data), cv::IMREAD_COLOR);
        cv::Mat img_left = cv::imdecode(cv::Mat(img_left_msg->data), cv::IMREAD_COLOR);
        std::vector<cv::Point3f> P3; for (auto& p : in->points) P3.emplace_back(p.x, p.y, p.z);

        colorize(P3, img_right, T_lidar_camera_right_, cv_K_right_, distCoeffs_right_, distortion_model_right_, width_right_, height_right_, true, false, out);
        colorize(P3, img_left, T_lidar_camera_left_, cv_K_left_, distCoeffs_left_, distortion_model_left_, width_left_, height_left_, false, true, out);

        out->width = out->points.size();
        sensor_msgs::PointCloud2 out_msg;
        pcl::toROSMsg(*out, out_msg); out_msg.header = cloud_msg->header;
        pub_.publish(out_msg);
    }

    void colorize(const std::vector<cv::Point3f>& P3, const cv::Mat& img, const Eigen::Matrix4d& T_camera_lidar,
        const cv::Mat& K, const cv::Mat& dist, const std::string& distortion_model,
        int width, int height, bool is_right, bool mirror_u,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& out) {

        // Invert the transform to get T_lidar_camera
        Eigen::Matrix4d T = T_camera_lidar.inverse();

        // Extract rotation and translation
        Eigen::Matrix3d R_e = T.block<3,3>(0,0);
        Eigen::Vector3d t_e = T.block<3,1>(0,3);
        cv::Mat R_cv, rvec, tvec(3,1,CV_64F);
        cv::eigen2cv(R_e, R_cv);
        cv::Rodrigues(R_cv, rvec);
        for (int i = 0; i < 3; ++i) tvec.at<double>(i,0) = t_e(i);

        // Project points
        std::vector<cv::Point2f> P2;
        if (distortion_model == "equidistant")
        cv::fisheye::projectPoints(P3, P2, rvec, tvec, K, dist);
        else
        cv::projectPoints(P3, P2, rvec, tvec, K, dist);

        for (size_t i = 0; i < P2.size(); ++i) {
        int u = static_cast<int>(std::round(P2[i].x));
        int v = static_cast<int>(std::round(P2[i].y));
        // if (mirror_u) u = width - 1 - u;
        if (u < 0 || u >= width || v < 0 || v >= height) continue;

        if ((is_right && P3[i].y <= 0.0) || (!is_right && P3[i].y >= 0.0)) {
            cv::Vec3b c = img.at<cv::Vec3b>(v, u);
            pcl::PointXYZRGB pt;
            pt.x = P3[i].x; pt.y = P3[i].y; pt.z = P3[i].z;
            pt.r = c[2]; pt.g = c[1]; pt.b = c[0];
            out->points.push_back(pt);
        }
        }
    }


    ros::NodeHandle nh_;
    std::string config_path_, cloud_topic_, output_topic_, image_topic_right_, image_topic_left_, distortion_model_right_, distortion_model_left_;
    Eigen::Matrix4d T_lidar_camera_right_, T_lidar_camera_left_;
    cv::Mat cv_K_right_, cv_K_left_, distCoeffs_right_, distCoeffs_left_;
    std::vector<double> intr_right_, intr_left_, dist_right_, dist_left_, resolution_right_, resolution_left_;
    int width_right_, height_right_, width_left_, height_left_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::CompressedImage> sub_img_right_, sub_img_left_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    ros::Publisher pub_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "colorize_pointcloud_node");
    ros::NodeHandle nh("~");
    PointCloudColorizer node(nh); ros::spin();
    return 0;
}