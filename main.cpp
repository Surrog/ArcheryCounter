#include "NN_approach.hpp"
#include "colour_filters.hpp"
#include "ellispes_approach.hpp"
#include "sources.hpp"
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr const char* model = IMAGE_TEST_DIR R".(\model\cleaned.jpg).";
constexpr const char* test = IMAGE_TEST_DIR R".(\20190325_204137.jpg).";

template <typename IT> void display_target_contours_masks(IT target_filled_begin, IT target_filled_end)
{
    while(target_filled_begin != target_filled_end)
    {
        cv::namedWindow("mask", cv::WINDOW_NORMAL);
        cv::imshow("mask", *target_filled_begin);
        cv::waitKey(0);
        cv::destroyAllWindows();
        ++target_filled_begin;
    }
}

std::array<cv::Mat, 10> compute_target_ring_mask(
    const std::array<cv::RotatedRect, 10>& target_ellipses, int img_row, int img_col, bool debug = false)
{
    std::array<std::vector<cv::Point>, 10> target_contours;
    std::transform(
        target_ellipses.begin(), target_ellipses.end(), target_contours.begin(), [](const cv::RotatedRect& rect) {
            std::vector<cv::Point> result;
            cv::Size2f size(rect.size.width / 2.f, rect.size.height / 2.f);
            cv::ellipse2Poly(rect.center, size, static_cast<int>(rect.angle), 0, 360, 1, result);
            return result;
        });

    std::array<cv::Mat, 10> target_ring_mask;
    std::transform(target_contours.begin(), target_contours.end(), target_ring_mask.begin(),
        [img_row, img_col](const std::vector<cv::Point>& contours_pts) {
            cv::Mat result = cv::Mat::zeros(img_row, img_col, CV_8U);
            cv::fillConvexPoly(result, contours_pts, 255);
            return result;
        });

    // display_target_contours_masks(target_ring_mask.begin(), target_ring_mask.end());
    return target_ring_mask;
}

std::size_t get_arrow_point(const cv::Point& tip_pos, const std::array<cv::Mat, 10> mask)
{
    std::size_t result = mask.size();

    for(std::size_t i = 0; i < mask.size() && mask[i].at<char>(tip_pos) == 0; i++)
    {
        result--;
    }
    return result;
}

int main(int argc, char** argv)
{
    cv::Mat image_test = cv::imread(test, cv::IMREAD_COLOR);

    if(!image_test.data)
    {
        printf("No image data\n");
        return -1;
    }

    cv::Mat image_test_gray;
    cv::cvtColor(image_test, image_test_gray, cv::COLOR_BGR2GRAY);

    std::array<cv::RotatedRect, 10> target_ellipses = ellipses::find_target(image_test, false);

    std::array<cv::Mat, 10> target_ring_mask
        = compute_target_ring_mask(target_ellipses, image_test.rows, image_test.cols);

    auto arrows_found = NN::find_arrows(image_test, image_test_gray, target_ring_mask);

    ellipses::display_ellipses(image_test, target_ellipses.begin(), target_ellipses.end());

    assert(arrows_found.tip.size() == arrows_found.fletching.size());

    std::vector<std::size_t> arrow_point(arrows_found.tip.size());
    std::transform(arrows_found.tip.begin(), arrows_found.tip.end(), arrow_point.begin(),
        [&target_ring_mask](const cv::Point& tip_pos) { return get_arrow_point(tip_pos, target_ring_mask); });

	for(std::size_t i = 0; i < arrow_point.size();                                                                       i++)
    {
        std::cout << "arrow " << i << arrow_point[i] << '\n';
    }

    return 0;
}