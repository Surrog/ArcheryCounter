#include "pretreatment.hpp"

cv::Mat pretreatment(const cv::Mat& img)
{
    cv::Mat pretreated = img.clone();
    cv::copyMakeBorder(pretreated, pretreated, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));

    cv::GaussianBlur(pretreated, pretreated, cv::Size(15, 15), 1.5, 1.5);
    cv::erode(pretreated, pretreated, cv::Mat());
    cv::dilate(pretreated, pretreated, cv::Mat(), cv::Point(-1, -1), 3);
    return pretreated;
}
