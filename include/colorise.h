#ifndef COLORISE_H
#define COLORISE_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CompressedImage.h>
#include <nav_msgs/Path.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <deque>
#include <limits>
#include <map>
#include <pcl/filters/filter.h>
#include <pcl/filters/impl/filter.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Imu.h>
#include <boost/optional.hpp>
#include <nav_msgs/Odometry.h>
#include <ros/package.h>



struct PointXYZRGBIntensity {
    PCL_ADD_POINT4D;
    float intensity;
    std::uint32_t t;
    std::uint16_t reflectivity;
    std::uint16_t ring;
    std::uint16_t ambient;
    std::uint32_t range;
    std::uint32_t rgb;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBIntensity,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (std::uint32_t, t, t)
                                (std::uint16_t, reflectivity, reflectivity)
                                (std::uint16_t, ring, ring)
                                (std::uint16_t, ambient, ambient)
                                (std::uint32_t, range, range)
                                (std::uint32_t, rgb, rgb)
)

struct PointKey {
    float x, y, z;
    bool operator<(const PointKey& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

class PointCloudColorizer {
public:
    explicit PointCloudColorizer(ros::NodeHandle& nh);

    void runMapNode();
    void saveFinalMap();
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void imgRightCallback(const sensor_msgs::CompressedImageConstPtr& msg);
    void imgLeftCallback(const sensor_msgs::CompressedImageConstPtr& msg);
    void trySyncAndProcess();
    void pathCallback(const nav_msgs::PathConstPtr& msg);
    void mapCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    bool lookupPose(const ros::Time& t, Eigen::Matrix4d& T_out);
    cv::Vec3b rgbToVec3b(uint32_t rgb);
    uint32_t vec3bToRgb(const cv::Vec3b& c);
    cv::Vec3b averageColor(const std::vector<cv::Vec3b>& colors);
    void colorizeFromPCD(const ros::TimerEvent&);

    // NEW: CameraInfo callbacks
    void camInfoRightCallback(const sensor_msgs::CameraInfoConstPtr& msg);
    void camInfoLeftCallback(const sensor_msgs::CameraInfoConstPtr& msg);

    void callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                const sensor_msgs::CompressedImageConstPtr& img_right_msg,
                const sensor_msgs::CompressedImageConstPtr& img_left_msg,
                const Eigen::Matrix4d& T_compensation);

    void colorize(const std::vector<cv::Point3f>& P3, const cv::Mat& img,
                  const Eigen::Matrix4d& T_camera_lidar, const cv::Mat& K, const cv::Mat& dist,
                  const std::string& distortion_model, int width, int height,
                  bool is_right, bool mirror_u,
                  pcl::PointCloud<PointXYZRGBIntensity>::Ptr& out,
                  const pcl::PointCloud<PointXYZRGBIntensity>::Ptr& in);

    template<typename T>
    typename T::value_type findClosest(const T& buffer, ros::Time target_time);

    template<typename T>
    void cleanOldMsgs(std::deque<T>& buffer, ros::Time latest_time);
    void imuCallback(const sensor_msgs::ImuConstPtr& msg);
    bool interpolateIMUOrientation(ros::Time t, Eigen::Quaterniond& q_out);
    void odomCallback(const nav_msgs::OdometryConstPtr& msg);
    bool interpolateOdometry(ros::Time t, Eigen::Matrix4d& T_out);
    Eigen::Matrix4d computeMotionCompensationOdom(ros::Time t_lidar, ros::Time t_cam);
    Eigen::Matrix4d computeMotionCompensation(ros::Time t_lidar, ros::Time t_cam);
    template<typename T>
    typename T::value_type findClosestBefore(const T& buffer, ros::Time reference_time);

private:
    ros::NodeHandle nh_;
    double max_time_offset_;
    double initial_startup_delay_;
    ros::Timer color_timer_;
    bool keep_uncolored_points_;
    double max_lidar_z_;
    std::string config_path_, cloud_topic_, output_topic_;
    std::string image_topic_right_, image_topic_left_;
    std::string imu_topic_;
    std::string imu_frame_;
    ros::Subscriber sub_imu_;
    std::deque<sensor_msgs::ImuConstPtr> imu_buffer_;
    std::string odom_compensation_frame_; // e.g. "base_link"
    ros::Subscriber sub_odom_;
    std::deque<nav_msgs::OdometryConstPtr> odom_buffer_;
    bool use_odom_compensation_;  // true=odom, false=imu

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string lidar_frame_;

    // NEW: CameraInfo subscribers and latched storage
    ros::Subscriber sub_info_right_, sub_info_left_;
    std::string camera_info_topic_right_, camera_info_topic_left_;
    boost::optional<sensor_msgs::CameraInfo> cam_info_right_, cam_info_left_;

    ros::Subscriber sub_cloud_, sub_img_right_, sub_img_left_;
    ros::Publisher pub_;

    std::deque<sensor_msgs::PointCloud2ConstPtr> cloud_buffer_;
    std::deque<sensor_msgs::CompressedImageConstPtr> img_right_buffer_;
    std::deque<sensor_msgs::CompressedImageConstPtr> img_left_buffer_;

    // For map node (unchanged)
    std::string map_topic_, odom_topic_, save_path_;
    std::string map_pcd_path_;
    int min_color_frames_;
    ros::Subscriber map_sub_, path_sub_;
    nav_msgs::Path latest_path_;
    pcl::PointCloud<PointXYZRGBIntensity>::Ptr accumulated_map_;
    std::map<PointKey, std::vector<cv::Vec3b>> color_history_;
    pcl::PointCloud<PointXYZRGBIntensity>::Ptr map_points_;
};


#endif  // COLORISE_H