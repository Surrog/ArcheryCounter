#include "ellispes_approach.hpp"
#include "math_utils.hpp"

#include <numeric>

std::vector<cv::Point> cleanup_center_points(std::vector<cv::Point> points, const cv::Mat& img, bool debug)
{
    std::ptrdiff_t erase_count = 0;
    cv::RotatedRect ellipse;
    std::size_t iteration = 0;
    const constexpr std::size_t max_iteration = 8;
    const constexpr double step = 0.03;
    const constexpr double lower_start = 0.48 - (step * double(max_iteration));
    const constexpr double upper_start = 0.58 + (step * double(max_iteration));
    while(iteration < 12 && points.size() > 5)
    {
        ellipse = cv::fitEllipse(points);

        if(debug)
        {
            auto tmp = img.clone();
            cv::drawContours(tmp, std::vector<std::vector<cv::Point>> {points}, 0, cv::Scalar(255, 0, 0), 3);
            cv::ellipse(tmp, ellipse, cv::Scalar(0, 255, 0), 3);
            cv::namedWindow("Cleanup", cv::WINDOW_NORMAL);
            cv::imshow("Cleanup", tmp);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        auto lower_threshold = ellipse.size.width * (lower_start + step * std::min(iteration, max_iteration));
        auto upper_threshold = ellipse.size.width * (upper_start - step * std::min(iteration, max_iteration));

        cv::Point center = ellipse.center;

        auto it = std::remove_if(
            points.begin(), points.end(), [center, lower_threshold, upper_threshold](const cv::Point& p) {
                auto dis = euclideanDistance(p, center);
                return dis < lower_threshold || dis > upper_threshold;
            });
        erase_count = std::distance(it, points.end());
        points.erase(it, points.end());
        iteration++;
    }
    return points;
}

cv::Point estimate_target_center(const cv::Mat& pretreat, bool debug)
{
    if(debug)
    {
        cv::namedWindow("gray", cv::WINDOW_NORMAL);
        cv::imshow("gray", pretreat);
        cv::waitKey(0);
    }

    std::vector<std::vector<cv::Point>> contours;
    findContours(pretreat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> result;

    for(auto& v : contours)
    {
        if(v.size() > result.size())
        {
            result = std::move(v);
        }
    }

    return cv::fitEllipse(result).center;
}

std::pair<std::vector<cv::Point>, cv::RotatedRect> detect_circle_approach(
    const cv::Mat& pretreat, const cv::Mat& original, bool agregate_contour, bool debug)
{
    if(debug)
    {
        cv::namedWindow("gray", cv::WINDOW_NORMAL);
        cv::imshow("gray", pretreat);
        cv::waitKey(0);
    }

    std::vector<std::vector<cv::Point>> contours;
    findContours(pretreat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> result;

    for(auto& v : contours)
    {
        if(v.size() > result.size())
        {
            result = v;
        }
    }

    if(agregate_contour)
    {
        auto top_center = center_of_mass(result);
        auto meand = mean_distance(top_center, result) * 2.5;

        for(auto& r : contours)
        {
            if(r.size() > 6)
            {
                auto local_center = center_of_mass(r);

                if(top_center != local_center && euclideanDistance(top_center, local_center) < meand)
                {
                    result.insert(result.end(), r.begin(), r.end());
                }
            }
        }

        result = cleanup_center_points(std::move(result), original, debug);
    }

    cv::RotatedRect rect;
    if(result.size() > 5)
        rect = cv::fitEllipse(result);

    return {result, rect};
}

std::size_t find_weakest_element(const std::vector<cv::RotatedRect>& ellipses)
{
    for(std::size_t i = 0; i < ellipses.size(); i++)
    {
        if(ellipses[i].center == cv::Point2f() && ellipses[i].size == cv::Size2f())
        {
            return i;
        }
    }

    std::vector<std::size_t> bad_score(ellipses.size(), 0);
    std::vector<double> distances(ellipses.size());

    std::vector<cv::Point2f> centers(ellipses.size());
    std::transform(
        ellipses.begin(), ellipses.end(), centers.begin(), [](const cv::RotatedRect& e) { return e.center; });
    auto mcenter = center_of_mass(centers);
    for(std::size_t i = 0; i < ellipses.size(); i++)
    {
        distances[i] = euclideanDistance(
            cv::Point2f(static_cast<float>(mcenter.x), static_cast<float>(mcenter.y)), ellipses[i].center);
    }
    auto max = std::minmax_element(distances.begin(), distances.begin() + 3).second;
    if(*max > 20)
        bad_score[std::distance(distances.begin(), max)]++;

    std::vector<float> angles(ellipses.size());
    std::transform(ellipses.begin(), ellipses.end(), angles.begin(), [](const cv::RotatedRect& e) { return e.angle; });
    auto mangles = std::accumulate(angles.begin(), angles.end(), 0.f) / angles.size();
    for(std::size_t i = 0; i < ellipses.size(); i++)
    {
        distances[i] = std::abs(mangles - ellipses[i].angle);
    }
    max = std::minmax_element(distances.begin(), distances.begin() + 3).second;
    bad_score[std::distance(distances.begin(), max)]++;

    std::vector<double> ratios(ellipses.size());
    std::transform(ellipses.begin(), ellipses.end(), ratios.begin(), [](const cv::RotatedRect& e) {
#if CV_MAJOR_VERSION < 3
        return e.size.width / static_cast<double>(e.size.height);
#else //! CV_MAJOR_VERSION > 3
            return e.size.aspectRatio();
#endif // CV_MAJOR_VERSION <> 3
    });
    auto mratios = std::accumulate(ratios.begin(), ratios.end(), 0.) / ratios.size();
    for(std::size_t i = 0; i < ellipses.size(); i++)
    {
#if CV_MAJOR_VERSION < 3
        double aspect = ellipses[i].size.width / static_cast<double>(ellipses[i].size.height);
#else //! CV_MAJOR_VERSION > 3
        double aspect = ellipses[i].size.aspectRatio();
#endif // CV_MAJOR_VERSION <> 3

        distances[i] = std::abs(mratios - aspect);
    }
    max = std::minmax_element(distances.begin(), distances.begin() + 3).second;
    if(*max > 0.01)
        bad_score[std::distance(distances.begin(), max)]++;

    auto max_bad = std::minmax_element(bad_score.begin(), bad_score.end()).second;
    if(*max_bad > 1)
        return std::distance(bad_score.begin(), max_bad);
    return bad_score.size();
}

std::array<cv::RotatedRect, 5> compute_final_ellipses(
    const std::array<cv::RotatedRect, 3>& ellipses, std::size_t ignored_index)
{
    std::array<cv::RotatedRect, 5> result = {};

    auto sample_size = ellipses.size();
    if(ignored_index < ellipses.size())
        sample_size--;

    double x_sum, x_x_sum, height_sum, x_height_sum, width_sum, x_width_sum, angle_sum, x_angle_sum, centerx_sum,
        x_centerx_sum, centery_sum, x_centery_sum;
    x_sum = x_x_sum = height_sum = x_height_sum = width_sum = x_width_sum = angle_sum = x_angle_sum = centerx_sum
        = x_centerx_sum = centery_sum = x_centery_sum = 0;

    for(std::size_t i = 0; i < ellipses.size(); i++)
    {
        if(i != ignored_index)
        {
            x_sum += i;
            x_x_sum += i * i;

            height_sum += ellipses[i].size.height;
            x_height_sum += i * double(ellipses[i].size.height);

            width_sum += ellipses[i].size.width;
            x_width_sum += i * double(ellipses[i].size.width);

            angle_sum += ellipses[i].angle;
            x_angle_sum += i * double(ellipses[i].angle);

            centerx_sum += ellipses[i].center.x;
            x_centerx_sum += i * double(ellipses[i].center.x);
            centery_sum += ellipses[i].center.y;
            x_centery_sum += i * double(ellipses[i].center.y);
        }
    }

    double x_square = (sample_size * x_x_sum - x_sum * x_sum);

    double height_coef = (sample_size * x_height_sum - x_sum * height_sum) / x_square;
    double height_constant = (height_sum - height_coef * x_sum) / sample_size;

    double width_coef = (sample_size * x_width_sum - x_sum * width_sum) / x_square;
    double width_constant = (width_sum - width_coef * x_sum) / sample_size;

    double angle_coef = (sample_size * x_angle_sum - x_sum * angle_sum) / x_square;
    double angle_constant = (angle_sum - angle_coef * x_sum) / sample_size;

    double centerx_coef = (sample_size * x_centerx_sum - x_sum * centerx_sum) / x_square;
    double centerx_constant = (centerx_sum - centerx_coef * x_sum) / sample_size;
    double centery_coef = (sample_size * x_centery_sum - x_sum * centery_sum) / x_square;
    double centery_constant = (centery_sum - centery_coef * x_sum) / sample_size;

    for(std::size_t i = 0; i < result.size(); i++)
    {
        cv::Point2f center(float(centerx_coef * i + centerx_constant), float(centery_coef * i + centery_constant));
        cv::Size2f size(float(width_coef * i + width_constant), float(height_coef * i + height_constant));
        float angle = float(angle_coef * i + angle_constant);
        result[i] = cv::RotatedRect(center, size, angle);
    }

    return result;
}
