#include "utils.h"


void parse(std::shared_ptr<cv::FileStorage> &config_file_storage, const std::string &parameter_name, Eigen::VectorXd &parsed_values){
    std::stringstream log_stream;
    
    cv::FileNode node = (*config_file_storage)[parameter_name];
    log_stream << parameter_name << " : [ ";

    for (size_t i = 0; i < node.size(); ++i) {
        parsed_values(i) = static_cast<double>(node[i]);
        log_stream << parsed_values(i) << ", ";
        
    }
    
    std::cout << log_stream.str() << "]" << std::endl;

}

void parse(std::shared_ptr<cv::FileStorage> &config_file_storage, const std::string &parameter_name, Eigen::Matrix4d &parsed_values){
    std::stringstream log_stream;
    cv::FileNode node = (*config_file_storage)[parameter_name];
    
    log_stream << parameter_name << " : [ " << "\n";

    for (size_t i = 0; i < node.size(); ++i) {
        cv::FileNode row = node[i];
        if (row.size() != 4) {
            std::cerr << "Error: Row " << i << " does not have 4 columns!" << std::endl;
            return;
        }

        for (size_t j = 0; j < row.size(); ++j) {
            parsed_values(i, j) = static_cast<double>(row[j]);

            log_stream << parsed_values(i) << ", ";
        }

        if(i < node.size()-1) {log_stream << "\n";}
    }

    std::cout << log_stream.str() << "]" << std::endl;


}

// convert keypoint from pixel image plane to camera coordinate
cv::Point2d camera2pixel(const cv::Point3d &pt3d, cv::Mat &K)
{
    return cv::Point2d(
        ( pt3d.x * K.at<double>(0,0) ) / pt3d.z + K.at<double>(0,2), 
        ( pt3d.y * K.at<double>(1,1) ) / pt3d.z + K.at<double>(1,2)
    );
}

