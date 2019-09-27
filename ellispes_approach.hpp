#ifndef ELLISPES_APPROACH_HPP
#define ELLISPES_APPROACH_HPP

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::Point> cleanup_center_points(std::vector<cv::Point> points, const cv::Mat& img, bool debug);

cv::Point estimate_target_center(const cv::Mat& pretreat, bool debug);

std::pair<std::vector<cv::Point>, cv::RotatedRect> detect_circle_approach(
    const cv::Mat& pretreat, const cv::Mat& original, bool agregate_contour = true, bool debug = false);

std::size_t find_weakest_element(const std::vector<cv::RotatedRect>& ellipses);

std::array<cv::RotatedRect, 5> compute_final_ellipses(
    const std::array<cv::RotatedRect, 3>& ellipses, std::size_t ignored_index);

#endif //! HISTO_APPROACH_HPP