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
        sensor_msgs::CompressedImage
    > SyncPolicy;

    PointCloudColorizer(ros::NodeHandle& nh) : nh_(nh) {
        nh_.param("config_path", config_path_, std::string("../configs/config.yaml"));
        cv::FileStorage fs(config_path_, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            ROS_ERROR("Unable to open %s", config_path_.c_str());
            ros::shutdown();
            return;
        }

        fs["image_topic"]      >> image_topic_;
        fs["pointcloud_topic"] >> cloud_topic_;
        fs["output_topic"]     >> output_topic_;

        std::vector<double> intr; fs["intrinsics"] >> intr;
        if (intr.size() != 4) {
            ROS_ERROR("intrinsics must be [fx, fy, cx, cy]");
            ros::shutdown();
            return;
        }
        cv_K_ = cv::Mat::eye(3, 3, CV_64F);
        cv_K_.at<double>(0,0) = intr[0]; cv_K_.at<double>(1,1) = intr[1];
        cv_K_.at<double>(0,2) = intr[2]; cv_K_.at<double>(1,2) = intr[3];

        std::vector<double> dist; fs["distortion_coeffs"] >> dist;
        distCoeffs_ = cv::Mat(dist);
        fs["distortion_model"] >> distortion_model_;

        fs["resolution"] >> resolution_;
        width_ = int(resolution_[0]); height_ = int(resolution_[1]);

        parseMat(fs, "T_lidar_camera", T_lidar_camera_);
        fs.release();

        sub_cloud_.subscribe(nh_, cloud_topic_, 1);
        sub_img_.subscribe(nh_, image_topic_, 1);
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), sub_cloud_, sub_img_));
        sync_->registerCallback(boost::bind(&PointCloudColorizer::callback, this, _1, _2));

        pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);
        ROS_INFO("Publishing colorized cloud on %s", output_topic_.c_str());
    }

private:
    void callback(
        const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
        const sensor_msgs::CompressedImageConstPtr& img_msg
    ) {
        // Convert cloud to PCL XYZ
        pcl::PointCloud<pcl::PointXYZ>::Ptr in(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *in);
        auto out = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        out->header.frame_id = cloud_msg->header.frame_id;
        out->is_dense = false;
        out->height = 1;

        // Decode compressed image
        cv::Mat img;
        try {
            img = cv::imdecode(cv::Mat(img_msg->data), cv::IMREAD_COLOR);
            if (img.empty()) {
                ROS_ERROR("Failed to decode compressed image");
                return;
            }
        } catch (const cv::Exception& e) {
            ROS_ERROR("cv::imdecode exception: %s", e.what());
            return;
        }

        // 1) Build P3 out of the *raw* LiDAR points
        std::vector<cv::Point3f> P3;
        P3.reserve(in->size());
        for (auto& p : in->points) {
        P3.emplace_back(p.x, p.y, p.z);
        }

        // 2) Compute rvec/tvec from T_lidar_camera (no inverse here!)
        Eigen::Matrix3d R_e = T_lidar_camera_.block<3,3>(0,0);
        Eigen::Vector3d t_e = T_lidar_camera_.block<3,1>(0,3);
        cv::Mat R_cv, rvec, tvec(3,1,CV_64F);
        cv::eigen2cv(R_e, R_cv);
        cv::Rodrigues(R_cv, rvec);
        for (int i = 0; i < 3; ++i) tvec.at<double>(i,0) = t_e(i);

        // 3) Project—all in one go
        std::vector<cv::Point2f> P2;
        if (distortion_model_ == "equidistant")
            cv::fisheye::projectPoints(P3, P2, rvec, tvec, cv_K_, distCoeffs_);
        else
            cv::projectPoints   (P3, P2, rvec, tvec, cv_K_, distCoeffs_);

        // Colorize
        for (size_t i = 0; i < P2.size(); ++i) {
            
            int u = int(std::round(P2[i].x));
            int v = int(std::round(P2[i].y));
            if (u < 0 || u >= width_ || v < 0 || v >= height_) continue;
            cv::Vec3b c = img.at<cv::Vec3b>(v, u);
            pcl::PointXYZRGB pt;
            pt.x = P3[i].x; pt.y = P3[i].y; pt.z = P3[i].z;
            pt.r = c[2]; pt.g = c[1]; pt.b = c[0];
            out->points.push_back(pt);
        }
        out->width = out->points.size();

        sensor_msgs::PointCloud2 out_msg;
        pcl::toROSMsg(*out, out_msg);
        out_msg.header = cloud_msg->header;
        pub_.publish(out_msg);
    }

    ros::NodeHandle nh_;
    std::string config_path_, image_topic_, cloud_topic_, output_topic_, distortion_model_;
    Eigen::Matrix4d T_lidar_camera_;
    cv::Mat cv_K_, distCoeffs_;
    std::vector<double> resolution_;
    int width_, height_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_;
    message_filters::Subscriber<sensor_msgs::CompressedImage> sub_img_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    ros::Publisher pub_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "colorize_pointcloud_node");
    ros::NodeHandle nh("~");
    PointCloudColorizer node(nh);
    ros::spin();
    return 0;
}
