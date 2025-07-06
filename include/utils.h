#pragma once
#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>


void parse(std::shared_ptr<cv::FileStorage> &config_file_storage, const std::string &parameter_name, Eigen::Matrix4d &parsed_values);
void parseMat(const cv::FileStorage& fs, const std::string& name, Eigen::Matrix4d& M);

cv::Point2d camera2pixel(const cv::Point3d &pt3d, cv::Mat &K);