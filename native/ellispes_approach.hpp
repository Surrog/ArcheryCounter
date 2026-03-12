#ifndef ELLISPES_APPROACH_HPP
#define ELLISPES_APPROACH_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ellipses
{
std::vector<cv::Point> cleanup_center_points(std::vector<cv::Point> points, const cv::Mat& img, bool debug);

cv::Point estimate_target_center(const cv::Mat& pretreat, bool debug);

std::pair<std::vector<cv::Point>, cv::RotatedRect> detect_circle_approach(
    const cv::Mat& pretreat, const cv::Mat& original, bool agregate_contour = true, bool debug = false);

std::size_t find_weakest_element(const std::vector<cv::RotatedRect>& ellipses);

std::array<cv::RotatedRect, 10> compute_final_ellipses_by_linear_interpolation(
    const std::array<cv::RotatedRect, 3>& ellipses, std::size_t ignored_index);

std::array<cv::RotatedRect, 10> find_target(const cv::Mat& img, bool debug);

template <typename IT>
void display_ellipses(const cv::Mat& image_test, IT ellipses_begin, IT ellipses_end)
{
    cv::Mat tmp = image_test.clone();
    while(ellipses_begin != ellipses_end)
    {
        cv::ellipse(tmp, *ellipses_begin, cv::Scalar(0, 255, 0), 3);
        ++ellipses_begin;
    }

    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::imshow("Output", tmp);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
}

#endif //! HISTO_APPROACH_HPP