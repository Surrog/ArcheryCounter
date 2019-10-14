#include "NN_approach.hpp"
#include "colour_filters.hpp"
#include "ellispes_approach.hpp"
#include "sources.hpp"
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr const char* model = IMAGE_TEST_DIR R".(\model\cleaned.jpg).";
constexpr const char* test = IMAGE_TEST_DIR R".(\20190325_204137.jpg).";

void display_ellipses(const cv::Mat& image_test, const std::array<cv::RotatedRect, 5>& ellipses)
{
    cv::Mat tmp = image_test.clone();
    for(const auto& e : ellipses)
    {
        cv::ellipse(tmp, e, cv::Scalar(0, 255, 0), 3);
    }

    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::imshow("Output", tmp);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void display_target_contours_masks(const std::array<cv::Mat, 5>& target_filled)
{
    for(const auto& mat : target_filled)
    {
        cv::namedWindow("mask", cv::WINDOW_NORMAL);
        cv::imshow("mask", mat);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

std::array<cv::Mat, 5> compute_target_ring_mask(
    const std::array<cv::RotatedRect, 5>& target_ellipses, int img_row, int img_col, bool debug = false)
{
    std::array<std::vector<cv::Point>, 5> target_contours;
    std::transform(
        target_ellipses.begin(), target_ellipses.end(), target_contours.begin(), [](const cv::RotatedRect& rect) {
            std::vector<cv::Point> result;
            cv::Size2f size(rect.size.width / 2.f, rect.size.height / 2.f);
            cv::ellipse2Poly(rect.center, size, static_cast<int>(rect.angle), 0, 360, 1, result);
            return result;
        });

    std::array<cv::Mat, 5> target_ring_mask;
    std::transform(target_contours.begin(), target_contours.end(), target_ring_mask.begin(),
        [img_row, img_col](const std::vector<cv::Point>& contours_pts) {
            cv::Mat result = cv::Mat::zeros(img_row, img_col, CV_8U);
            cv::fillConvexPoly(result, contours_pts, 255);
            return result;
        });

    // Here we remove the inner circle of the previous ring
    for(std::size_t i = 1; i < target_ring_mask.size(); i++)
    {
        cv::fillConvexPoly(target_ring_mask[i], target_contours[i - 1], 0);
    }

    // display_target_contours_masks(target_ring_mask);
    return target_ring_mask;
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

    std::array<cv::RotatedRect, 5> target_ellipses = find_target_ellipses(image_test, false);

    std::array<cv::Mat, 5> target_ring_mask
        = compute_target_ring_mask(target_ellipses, image_test.rows, image_test.cols);

    auto arrows_found = NN::find_arrows(image_test, image_test_gray, target_ring_mask);

    display_ellipses(image_test.clone(), target_ellipses);

    return 0;
}