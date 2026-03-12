#ifndef KEYPOINT_APPROACH_HPP
#define KEYPOINT_APPROACH_HPP

#include <opencv2/opencv.hpp>

struct ORB_Param
{
    int nfeature = 500;
    int nlevels = 8;
    int edgeThreshold = 31;
    int fastThreshold = 20;
    int patchSize = 31;
    int WTA_K = 2;

    enum
    {
        NLEVEL_MIN = 1,
        NLEVEL_MAX = 16,
        EDGE_THRESHOLD_MIN = 1,
        EDGE_THRESHOLD_MAX = 500,
        FAST_THRESHOLD_MIN = 1,
        FAST_THRESHOLD_MAX = 500,
        PATCH_SIZE_MIN = 5,
        PATCH_SIZE_MAX = 150
    };
};

std::vector<cv::DMatch> filter_match(const std::vector<cv::KeyPoint>& keypoint_model,
    const std::vector<cv::KeyPoint>& keypoint_test, const std::vector<std::vector<cv::DMatch>>& matches,
    double lowe_ratio);

std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoint_compute(const cv::Mat& img, const ORB_Param& param);

std::size_t test_orb_param(const cv::Mat& image_model, const cv::Mat& image_test, const ORB_Param& val);

void keypoint_approach(const cv::Mat& image_model, const cv::Mat& image_test);

#endif // !KEYPOINT_APPROACH_HPP