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

    // -----------------------------------------------------------------------
    // Map node
    // -----------------------------------------------------------------------
    void runMapNode();
    void saveFinalMap();
    void pathCallback(const nav_msgs::PathConstPtr& msg);
    void mapCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    bool lookupPose(const ros::Time& t, Eigen::Matrix4d& T_out);
    cv::Vec3b rgbToVec3b(uint32_t rgb);
    uint32_t vec3bToRgb(const cv::Vec3b& c);
    cv::Vec3b averageColor(const std::vector<cv::Vec3b>& colors);
    void colorizeFromPCD(const ros::TimerEvent&);

    // -----------------------------------------------------------------------
    // Scan colorisation
    // -----------------------------------------------------------------------
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    void imgRightCallback(const sensor_msgs::CompressedImageConstPtr& msg);
    void imgLeftCallback(const sensor_msgs::CompressedImageConstPtr& msg);
    void trySyncAndProcess();

    void camInfoRightCallback(const sensor_msgs::CameraInfoConstPtr& msg);
    void camInfoLeftCallback(const sensor_msgs::CameraInfoConstPtr& msg);

    // Main processing callback — timestamps passed explicitly so each camera
    // can be compensated to its own capture time.
    void callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                  const sensor_msgs::CompressedImageConstPtr& img_right_msg,
                  const sensor_msgs::CompressedImageConstPtr& img_left_msg,
                  ros::Time t_lidar,
                  ros::Time t_cam);

    // Project P3 (lidar-frame points at t_lidar) into img using the unified
    // T_cam_from_lidar transform (motion-compensated + extrinsic in one step).
    // Project P3 into img using T_cam_from_points (motion-compensated + extrinsic).
    // source_z_max: filters points by their Z in the SOURCE frame before projection.
    //   ColoriseScan: pass max_lidar_z_ (lidar-frame height filter).
    //   ColoriseMap:  omit or pass FLT_MAX — world-frame Z is not meaningful here,
    //                 the camera-depth check (pt_cam.z > 0) handles occlusion instead.
    void colorize(const std::vector<cv::Point3f>& P3,
                  const cv::Mat& img,
                  const Eigen::Matrix4d& T_cam_from_points,
                  const cv::Mat& K,
                  const cv::Mat& dist,
                  const std::string& distortion_model,
                  int width, int height,
                  pcl::PointCloud<PointXYZRGBIntensity>::Ptr& out,
                  const pcl::PointCloud<PointXYZRGBIntensity>::Ptr& in,
                  float source_z_max = std::numeric_limits<float>::max());

    // -----------------------------------------------------------------------
    // Motion compensation — odometry path
    //
    // Computes the full T_cam(t_cam) <- lidar(t_lidar) transform by
    // interpolating odometry at both timestamps and composing with static TF
    // extrinsics for both the lidar and the named camera frame.
    // Returns false if odometry or TF data are unavailable.
    // -----------------------------------------------------------------------
    bool computeCamFromLidar(ros::Time t_lidar, ros::Time t_cam,
                             const std::string& cam_frame,
                             Eigen::Matrix4d& T_out);

    // -----------------------------------------------------------------------
    // Motion compensation — IMU path
    //
    // Computes T_cam(t_cam) <- lidar(t_lidar) using AHRS SLERP for rotation
    // and double-integrated gravity-removed accelerometer for translation,
    // then composes with the static camera extrinsic from TF.
    // Returns false if IMU data or TF extrinsics are unavailable.
    // -----------------------------------------------------------------------
    bool computeCamFromLidarIMU(ros::Time t_lidar, ros::Time t_cam,
                                const std::string& cam_frame,
                                Eigen::Matrix4d& T_out);

    // -----------------------------------------------------------------------
    // IMU / odometry helpers
    // -----------------------------------------------------------------------
    void imuCallback(const sensor_msgs::ImuConstPtr& msg);
    bool interpolateIMUOrientation(ros::Time t, Eigen::Quaterniond& q_out);

    void odomCallback(const nav_msgs::OdometryConstPtr& msg);
    bool interpolateOdometry(ros::Time t, Eigen::Matrix4d& T_out);

    // -----------------------------------------------------------------------
    // Buffer utilities
    // -----------------------------------------------------------------------
    template<typename T>
    typename T::value_type findClosest(const T& buffer, ros::Time target_time);

    template<typename T>
    typename T::value_type findClosestBefore(const T& buffer, ros::Time reference_time);

    template<typename T>
    void cleanOldMsgs(std::deque<T>& buffer, ros::Time latest_time);

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
    std::string base_frame_;              // robot body frame in TF tree (e.g. "base_link")
    ros::Subscriber sub_odom_;
    std::deque<nav_msgs::OdometryConstPtr> odom_buffer_;
    bool use_odom_compensation_;  // true=odom, false=imu

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string lidar_frame_;

    ros::Subscriber sub_info_right_, sub_info_left_;
    std::string camera_info_topic_right_, camera_info_topic_left_;
    boost::optional<sensor_msgs::CameraInfo> cam_info_right_, cam_info_left_;

    ros::Subscriber sub_cloud_, sub_img_right_, sub_img_left_;
    ros::Publisher pub_;

    std::deque<sensor_msgs::PointCloud2ConstPtr> cloud_buffer_;
    std::deque<sensor_msgs::CompressedImageConstPtr> img_right_buffer_;
    std::deque<sensor_msgs::CompressedImageConstPtr> img_left_buffer_;

    // -----------------------------------------------------------------------
    // Calibration extrinsics (loaded directly from calibration.yaml)
    // -----------------------------------------------------------------------
    bool            calib_loaded_      = false;
    std::string     calibration_path_;
    Eigen::Matrix4d T_cam_lidar_left_  = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_cam_lidar_right_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d R_imu_lidar_       = Eigen::Matrix3d::Identity();

    // -----------------------------------------------------------------------
    // Map node members
    // -----------------------------------------------------------------------
    std::string map_topic_, odom_topic_, save_path_;
    std::string map_pcd_path_;
    int min_color_frames_;
    ros::Subscriber map_sub_, path_sub_;
    nav_msgs::Path latest_path_;
    pcl::PointCloud<PointXYZRGBIntensity>::Ptr accumulated_map_;
    std::map<PointKey, std::vector<cv::Vec3b>> color_history_;
    pcl::PointCloud<PointXYZRGBIntensity>::Ptr map_points_;
    ros::Publisher pub_raw_map_;
    ros::Publisher pub_progress_;
    ros::Publisher pub_robot_pose_;
    ros::Publisher pub_frustum_;
    void publishRawMap();
    void publishRobotPose(const Eigen::Matrix4d& T_world_base, ros::Time t);
    void publishFrustum(const Eigen::Matrix4d& T_world_camera,
                        const sensor_msgs::CameraInfo& info,
                        ros::Time t, int id, float r, float g, float b);
};


#endif  // COLORISE_H