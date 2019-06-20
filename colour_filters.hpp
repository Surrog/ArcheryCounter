#ifndef COLOUR_FILTERS_HPP
#define COLOUR_FILTERS_HPP

#include <opencv2/opencv.hpp>

cv::Mat filter_white(const cv::Mat& img);

cv::Mat filter_yellow(const cv::Mat& img);

cv::Mat filter_red(const cv::Mat& img);

cv::Mat filter_blue(const cv::Mat& img);

cv::Mat filter_black(const cv::Mat& img);

cv::Mat filter_arrow(const cv::Mat& img, const std::array<cv::Mat, 5>& target_filled, bool debug = false);

#endif //COLOUR_FILTERS_HPP