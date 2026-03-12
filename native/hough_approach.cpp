#include "hough_approach.hpp"

std::vector<cv::Vec4i> hough_approach(const cv::Mat& img, bool debug)
{
    double hysteresis_threshold1 = 0;
    double hysteresis_threshold2 = 0;
    int aperture_size = 5;
    bool gradient = false;
    cv::Mat canny_output;

    cv::Canny(img, canny_output, hysteresis_threshold1, hysteresis_threshold2, aperture_size, gradient);

    if(debug)
    {
        cv::namedWindow("Canny", cv::WINDOW_NORMAL);
        cv::imshow("Canny", canny_output);
        cv::waitKey(0);
        cv::destroyAllWindows();

        // cv::Mat color_convert;
        // cv::cvtColor(canny_output, color_convert, cv::COLOR_GRAY2BGR);

        // cv::namedWindow("gray", cv::WINDOW_NORMAL);
        // cv::imshow("gray", color_convert);
        // cv::waitKey(0);
        // cv::destroyAllWindows();
    }

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(canny_output, lines, 1, CV_PI / 180, 100, 0, 0);

    return lines;
}
