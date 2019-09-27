#include "colour_filters.hpp"
#include "ellispes_approach.hpp"
#include "sources.hpp"
#include <algorithm>
#include <future>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr const char* model = IMAGE_TEST_DIR R".(\model\cleaned.jpg).";
constexpr const char* test = IMAGE_TEST_DIR R".(\20190325_204137.jpg).";

cv::Mat pretreatment(const cv::Mat& img)
{
    cv::Mat pretreated = img.clone();
    cv::copyMakeBorder(pretreated, pretreated, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));

    cv::GaussianBlur(pretreated, pretreated, cv::Size(15, 15), 1.5, 1.5);
    cv::erode(pretreated, pretreated, cv::Mat());
    cv::dilate(pretreated, pretreated, cv::Mat(), cv::Point(-1, -1), 3);
    return pretreated;
}

std::array<cv::Mat, 5> filter_image(const cv::Mat& pretreat)
{
    std::array<cv::Mat, 5> result;
    std::array<std::future<cv::Mat>, 5> async_result {std::async(filter_yellow, pretreat),
        std::async(filter_red, pretreat), std::async(filter_blue, pretreat), std::async(filter_black, pretreat),
        std::async(filter_white, pretreat)};

    for(std::size_t i = 0; i < result.size(); i++)
    {
        result[i] = async_result[i].get();
    }
    return result;
}

std::array<cv::RotatedRect, 5> find_target(const cv::Mat& img, bool debug)
{
    cv::Mat pretreat = pretreatment(img);
    std::array<cv::Mat, 5> filtered_images = filter_image(pretreat);
    std::array<bool, 5> agregate_contour {true, true, true, true, false};
    std::array<bool, 5> debug_contour {false, false, false, false, false};

    std::vector<std::future<std::pair<std::vector<cv::Point>, cv::RotatedRect>>> async_ellipse_process;
    async_ellipse_process.reserve(filtered_images.size());
    for(std::size_t i = 0; i < 3; i++)
    {
        if(debug_contour[i])
        {
            async_ellipse_process.emplace_back(std::async(
                std::launch::deferred, detect_circle_approach, filtered_images[i], img, agregate_contour[i], true));
        }
        else
        {
            async_ellipse_process.emplace_back(
                std::async(detect_circle_approach, filtered_images[i], img, agregate_contour[i], false));
        }
    }

    std::vector<cv::RotatedRect> ellipses;
    std::vector<std::vector<cv::Point>> contours;
    ellipses.reserve(async_ellipse_process.size());
    contours.reserve(async_ellipse_process.size());
    for(auto& async_p : async_ellipse_process)
    {
        auto pair = async_p.get();
        if(pair.first.size())
        {
            ellipses.emplace_back(std::move(pair.second));
            contours.emplace_back(std::move(pair.first));
        }
    }

    if(debug)
    {
        auto tmp = img.clone();
        cv::drawContours(tmp, contours, -1, cv::Scalar(255, 0, 0), 3);
        unsigned char shade = 32;
        for(const auto& e : ellipses)
        {
            cv::circle(tmp, e.center, 10, cv::Scalar(0, 0, shade), 6);
            cv::ellipse(tmp, e, cv::Scalar(0, 255, 0), 3);
            shade += 32;
        }

        cv::namedWindow("Output", cv::WINDOW_NORMAL);
        cv::imshow("Output", tmp);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    auto value = find_weakest_element(ellipses);

    if(debug)
    {
        std::cout << "weakest " << value << '\n';
    }

    return compute_final_ellipses({ellipses[0], ellipses[1], ellipses[2]}, value);
}

int main(int argc, char** argv)
{
    cv::Mat image_model = cv::imread(model, cv::IMREAD_COLOR);
    cv::Mat image_test = cv::imread(test, cv::IMREAD_COLOR);

    if(!image_model.data || !image_test.data)
    {
        printf("No image data\n");
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(image_test, gray, cv::COLOR_BGR2GRAY);

    std::cout << "gray type : " << gray.type() << '\n';
    auto target = find_target(image_test, false);

    std::array<std::vector<cv::Point>, 5> target_contours;
    std::transform(target.begin(), target.end(), target_contours.begin(), [](const cv::RotatedRect& rect) {
        std::vector<cv::Point> result;
        cv::Size2f size(rect.size.width / 2.f, rect.size.height / 2.f);
        cv::ellipse2Poly(rect.center, size, static_cast<int>(rect.angle), 0, 360, 1, result);
        return result;
    });

    // cv::Mat img_contours = image_test.clone();
    // cv::drawContours(img_contours, std::vector<std::vector<cv::Point>> {target_contours.begin(),
    // target_contours.end()},
    //    -1, cv::Scalar(0, 255, 0), 3);
    // cv::namedWindow("contours", cv::WINDOW_NORMAL);
    // cv::imshow("contours", img_contours);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    std::array<cv::Mat, 5> target_filled;
    std::transform(target_contours.begin(), target_contours.end(), target_filled.begin(),
        [img_row = image_test.rows, img_col = image_test.cols](const std::vector<cv::Point>& pts) {
            cv::Mat result = cv::Mat::zeros(img_row, img_col, CV_8U);
            cv::fillConvexPoly(result, pts, 255);
            return result;
        });

    for(std::size_t i = 1; i < target_filled.size(); i++)
    {
        cv::fillConvexPoly(target_filled[i], target_contours[i - 1], 0);
    }

    // for(const auto& mat : target_filled)
    //{
    //    cv::namedWindow("mask", cv::WINDOW_NORMAL);
    //    cv::imshow("mask", mat);
    //    cv::waitKey(0);
    //    cv::destroyAllWindows();
    //}

    auto arrow_mask = filter_arrow(image_test, target_filled);
    cv::Mat filter_gray;
    cv::bilateralFilter(gray, filter_gray, 7, 50, 50);
    cv::Mat edges;
    cv::Canny(filter_gray, edges, 70, 150, 3, false);
    cv::namedWindow("arrow", cv::WINDOW_NORMAL);
    cv::imshow("arrow", arrow_mask);
    cv::namedWindow("Canny", cv::WINDOW_NORMAL);
    cv::imshow("Canny", edges);
    cv::Mat filtered_edges = edges.clone();
    cv::bitwise_and(edges, arrow_mask, filtered_edges);
    cv::namedWindow("filtered Canny", cv::WINDOW_NORMAL);
    cv::imshow("filtered Canny", filtered_edges);

    cv::waitKey(0);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(filtered_edges, lines, 1, CV_PI / 360, 150, 150, 20);

    auto display_line = image_test.clone();
    for(const auto& l : lines)
    {
        cv::line(display_line, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3,
#if CV_MAJOR_VERSION < 3
            CV_AA
#else // CV_MAJOR_VERSION >= 3
            cv::LINE_AA
#endif // CV_MAJOR_VERSION <>= 3
        );
    }
    cv::namedWindow("line detected", cv::WINDOW_NORMAL);
    cv::imshow("line detected", display_line);

    cv::waitKey(0);

    cv::Mat tmp = image_test.clone();
    for(const auto& e : target)
    {
        cv::ellipse(tmp, e, cv::Scalar(0, 255, 0), 3);
    }

    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::imshow("Output", tmp);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}