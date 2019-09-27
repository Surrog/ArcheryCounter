#ifndef HISTO_APPROACH_HPP
#define HISTO_APPROACH_HPP

#include <cstdio>
#include <opencv2/opencv.hpp>

struct bound
{
    std::size_t ibegin = 0;
    int best_max = 0;
    std::size_t iend = 0;
    int best_min = 0;

    int quality() const { return best_max + -best_min; }
};

bound find_histo_bound(const std::vector<int>& histo);

struct line_bound
{
    std::size_t ibegin;
    std::size_t iend;
    std::size_t size;
};

line_bound find_longest_line(const cv::Mat& img);

std::vector<std::array<int, 4>> histogram_approach(const cv::Mat& img);

#endif //! HISTO_APPROACH_HPP