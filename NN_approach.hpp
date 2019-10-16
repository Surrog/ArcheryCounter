#ifndef NN_APPROACH_HPP
#define NN_APPROACH_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace NN
{
struct arrows_position
{
    std::vector<cv::Point> tip;
    std::vector<cv::Point> fletching;
};

arrows_position find_arrows(
    const cv::Mat& image_test, const cv::Mat& image_test_gray, const std::array<cv::Mat, 10>& target_ring_mask)
{

    return {};
}
}

#endif //! NN_APPROACH_HPP