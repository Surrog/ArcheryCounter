#include "colour_filters.hpp"
#include <future>

cv::Mat filter_white(const cv::Mat& img)
{
    cv::Mat pretreat = img.clone();
    cv::cvtColor(pretreat, pretreat, cv::COLOR_RGB2HSV_FULL);
    cv::inRange(pretreat, cv::Scalar(0, 0, 0), cv::Scalar(220, 25, 255), pretreat);
    return pretreat;
}

cv::Mat filter_yellow(const cv::Mat& img)
{
    cv::Mat pretreat = img.clone();
    cv::cvtColor(pretreat, pretreat, cv::COLOR_RGB2HSV_FULL);
    cv::inRange(pretreat, cv::Scalar(136, 64, 0), cv::Scalar(140, 255, 255), pretreat);
    return pretreat;
}

cv::Mat filter_red(const cv::Mat& img)
{
    cv::Mat pretreat = img.clone();
    cv::cvtColor(pretreat, pretreat, cv::COLOR_RGB2HSV_FULL);
    cv::inRange(pretreat, cv::Scalar(168, 64, 0), cv::Scalar(171, 255, 255), pretreat);
    return pretreat;
}

cv::Mat filter_blue(const cv::Mat& img)
{
    cv::Mat pretreat = img.clone();
    cv::cvtColor(pretreat, pretreat, cv::COLOR_RGB2HSV_FULL);
    cv::inRange(pretreat, cv::Scalar(24, 64, 0), cv::Scalar(32, 255, 255), pretreat);
    return pretreat;
}

cv::Mat filter_black(const cv::Mat& img)
{
    cv::Mat pretreat = img.clone();
    cv::cvtColor(pretreat, pretreat, cv::COLOR_RGB2HSV_FULL);
    cv::inRange(pretreat, cv::Scalar(128, 16, 32), cv::Scalar(192, 52, 96), pretreat);
    return pretreat;
}

cv::Mat filter_arrow(const cv::Mat& img, const std::array<cv::Mat, 5>& target_filled, bool debug)
{
    std::array<cv::Mat, 4> colours {
        255 - filter_yellow(img), 255 - filter_red(img), 255 - filter_blue(img), img.clone()};
    cv::cvtColor(colours[3], colours[3], cv::COLOR_RGB2HSV_FULL);
    cv::inRange(colours[3], cv::Scalar(0, 0, 0), cv::Scalar(255, 140, 80), colours[3]);

    if(debug)
    {
        for(const auto& mat : colours)
        {
            cv::namedWindow("black_layer", cv::WINDOW_NORMAL);
            cv::imshow("black_layer", mat);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
    assert(colours.size() > 3 && target_filled.size() >= colours.size());
    for(std::size_t i = 0; i < 3; i++)
    {
        cv::bitwise_and(colours[i], target_filled[i], colours[i]);
    }

    if(debug)
    {
        for(const auto& mat : colours)
        {
            cv::namedWindow("cleaned black_layer", cv::WINDOW_NORMAL);
            cv::imshow("cleaned black_layer", mat);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

    for(std::size_t i = 1; i < colours.size(); i++)
    {
        cv::bitwise_or(colours[0], colours[i], colours[0]);
    }

    cv::erode(colours[0], colours[0], cv::Mat(), cv::Point(-1, -1), 4);
    cv::dilate(colours[0], colours[0], cv::Mat(8, 8, 0), cv::Point(-1, -1), 8);

    return colours[0];
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
