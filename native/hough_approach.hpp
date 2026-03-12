#ifndef HOUGH_APPROACH_HPP
#define HOUGH_APPROACH_HPP

#include <opencv2/opencv.hpp>

std::vector<cv::Vec4i> hough_approach(const cv::Mat& img, bool debug = false);

#endif // !KEYPOINT_APPROACH_HPP