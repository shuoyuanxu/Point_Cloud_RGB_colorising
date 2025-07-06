#include "utils.h"

// convert keypoint from pixel image plane to camera coordinate
cv::Point2d camera2pixel(const cv::Point3d &pt3d, cv::Mat &K)
{
    return cv::Point2d(
        ( pt3d.x * K.at<double>(0,0) ) / pt3d.z + K.at<double>(0,2), 
        ( pt3d.y * K.at<double>(1,1) ) / pt3d.z + K.at<double>(1,2)
    );
}

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