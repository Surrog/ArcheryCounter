#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

template <typename PointT> inline auto euclideanDistance(const PointT& lval, const PointT& rval)
{
    auto diff = lval - rval;
    return cv::sqrt((diff.x * diff.x) + (diff.y * diff.y));
}

template <typename T> inline cv::Point center_of_mass(const T& list)
{
    double mean_x = 0;
    double mean_y = 0;
    for(const auto& p : list)
    {
        mean_x += double(p.x) / list.size();
        mean_y += double(p.y) / list.size();
    }
    return cv::Point(int(mean_x), int(mean_y));
}

inline auto mean_distance(const cv::Point& p, const std::vector<cv::Point>& list)
{
    double dis = 0;
    for(const auto& v : list)
    {
        dis += euclideanDistance(p, v);
    }
    return dis / list.size();
}

#endif // MATH_UTILS_HPP