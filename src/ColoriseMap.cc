// Modified PointCloudColorizer with initial delay and event-based sync (no timer loop)

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CompressedImage.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <deque>
#include <limits>

struct PointXYZRGBRing {
    PCL_ADD_POINT4D;
    std::uint32_t rgb;
    std::uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBRing,
                                  (float, x, x)(float, y, y)(float, z, z)
                                  (float, rgb, rgb)(std::uint16_t, ring, ring))

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
    PointCloudColorizer(ros::NodeHandle& nh) : nh_(nh) {
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

private:
    double max_time_offset_;
    double initial_startup_delay_;

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        cloud_buffer_.push_back(msg);
        cleanOldMsgs(cloud_buffer_, msg->header.stamp);
        trySyncAndProcess();
    }

    void imgRightCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
        img_right_buffer_.push_back(msg);
        cleanOldMsgs(img_right_buffer_, msg->header.stamp);
    }

    void imgLeftCallback(const sensor_msgs::CompressedImageConstPtr& msg) {
        img_left_buffer_.push_back(msg);
        cleanOldMsgs(img_left_buffer_, msg->header.stamp);
    }

    void trySyncAndProcess() {
        if (cloud_buffer_.empty() || img_right_buffer_.empty() || img_left_buffer_.empty()) return;

        for (auto it_cloud = cloud_buffer_.begin(); it_cloud != cloud_buffer_.end(); ++it_cloud) {
            ros::Time t = (*it_cloud)->header.stamp;
            auto img_right = findClosest(img_right_buffer_, t);
            auto img_left  = findClosest(img_left_buffer_, t);
            if (!img_right || !img_left) continue;

            callback(*it_cloud, img_right, img_left);
            cloud_buffer_.erase(cloud_buffer_.begin(), it_cloud + 1);
            return;
        }
    }

    template<typename T>
    typename T::value_type findClosest(const T& buffer, ros::Time target_time) {
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
    void cleanOldMsgs(std::deque<T>& buffer, ros::Time latest_time) {
        while (!buffer.empty() && (latest_time - buffer.front()->header.stamp).toSec() > 2.0)
            buffer.pop_front();
    }

    void callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                  const sensor_msgs::CompressedImageConstPtr& img_right_msg,
                  const sensor_msgs::CompressedImageConstPtr& img_left_msg) {

        pcl::PointCloud<PointXYZRGBRing>::Ptr in(new pcl::PointCloud<PointXYZRGBRing>);
        pcl::fromROSMsg(*cloud_msg, *in);
        auto out = boost::make_shared<pcl::PointCloud<PointXYZRGBRing>>();
        out->header.frame_id = cloud_msg->header.frame_id;
        out->is_dense = false; out->height = 1;

        cv::Mat img_right = cv::imdecode(cv::Mat(img_right_msg->data), cv::IMREAD_COLOR);
        cv::Mat img_left  = cv::imdecode(cv::Mat(img_left_msg->data),  cv::IMREAD_COLOR);
        std::vector<cv::Point3f> P3; 
        for (auto& p : in->points) P3.emplace_back(p.x, p.y, p.z);
        ROS_INFO_STREAM_THROTTLE(2.0, "[Sync Info] LiDAR time: " << cloud_msg->header.stamp
            << ", Right image time: " << img_right_msg->header.stamp
            << ", Left image time: " << img_left_msg->header.stamp);

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

    void colorize(const std::vector<cv::Point3f>& P3, const cv::Mat& img, 
                  const Eigen::Matrix4d& T_camera_lidar, const cv::Mat& K, const cv::Mat& dist, 
                  const std::string& distortion_model, int width, int height, 
                  bool is_right, bool mirror_u, 
                  pcl::PointCloud<PointXYZRGBRing>::Ptr& out,
                  const pcl::PointCloud<PointXYZRGBRing>::Ptr& in) {

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
            PointXYZRGBRing pt;
            pt.x = P3[i].x; pt.y = P3[i].y; pt.z = P3[i].z;
            pt.rgb = (static_cast<uint32_t>(c[2]) << 16 |
                      static_cast<uint32_t>(c[1]) << 8  |
                      static_cast<uint32_t>(c[0]));
            pt.ring = in->points[i].ring;
            out->points.push_back(pt);
        }
    }

    ros::NodeHandle nh_;
    std::string config_path_, cloud_topic_, output_topic_, image_topic_right_, image_topic_left_;
    std::string distortion_model_right_, distortion_model_left_;
    Eigen::Matrix4d T_lidar_camera_right_, T_lidar_camera_left_;
    cv::Mat cv_K_right_, cv_K_left_, distCoeffs_right_, distCoeffs_left_;
    std::vector<double> intr_right_, intr_left_, dist_right_, dist_left_, resolution_right_, resolution_left_;
    int width_right_, height_right_, width_left_, height_left_;

    ros::Subscriber sub_cloud_, sub_img_right_, sub_img_left_;
    ros::Publisher pub_;

    std::deque<sensor_msgs::PointCloud2ConstPtr> cloud_buffer_;
    std::deque<sensor_msgs::CompressedImageConstPtr> img_right_buffer_;
    std::deque<sensor_msgs::CompressedImageConstPtr> img_left_buffer_;
};

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
