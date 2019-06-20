#include "colour_filters.hpp"
#include "sources.hpp"
#include <algorithm>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <stdio.h>

constexpr const char* model = IMAGE_TEST_DIR R".(\model\cleaned.jpg).";
constexpr const char* test = IMAGE_TEST_DIR R".(\20190325_204137.jpg).";

template <typename PointT> auto euclideanDistance(const PointT& lval, const PointT& rval)
{
    auto diff = lval - rval;
    return cv::sqrt((diff.x * diff.x) + (diff.y * diff.y));
}

std::vector<cv::DMatch> filter_match(const std::vector<cv::KeyPoint>& keypoint_model,
    const std::vector<cv::KeyPoint>& keypoint_test, const std::vector<std::vector<cv::DMatch>>& matches,
    double lowe_ratio)
{
    std::vector<cv::DMatch> result;
    for(const auto& vals : matches)
    {
        if(vals[0].distance < lowe_ratio * vals[1].distance)
        {
            result.push_back(vals[0]);
        }
    }
    return result;
}

cv::Mat pretreatment(const cv::Mat& img)
{
    cv::Mat pretreated = img.clone();
    cv::copyMakeBorder(pretreated, pretreated, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));

    cv::GaussianBlur(pretreated, pretreated, cv::Size(15, 15), 1.5, 1.5);
    cv::erode(pretreated, pretreated, cv::Mat());
    cv::dilate(pretreated, pretreated, cv::Mat(), cv::Point(-1, -1), 3);
    return pretreated;
}

struct ORB_Param
{
    int nfeature = 500;
    int nlevels = 8;
    int edgeThreshold = 31;
    int fastThreshold = 20;
    int patchSize = 31;
    int WTA_K = 2;

    enum
    {
        NLEVEL_MIN = 1,
        NLEVEL_MAX = 16,
        EDGE_THRESHOLD_MIN = 1,
        EDGE_THRESHOLD_MAX = 500,
        FAST_THRESHOLD_MIN = 1,
        FAST_THRESHOLD_MAX = 500,
        PATCH_SIZE_MIN = 5,
        PATCH_SIZE_MAX = 150
    };
};

std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoint_compute(const cv::Mat& img, const ORB_Param& param)
{
#if CV_MAJOR_VERSION < 3
    auto orb = std::make_unique<cv::ORB>(param.nfeature, 1.2f, param.nlevels, param.edgeThreshold, 0, param.WTA_K,
        cv::ORB::HARRIS_SCORE, param.patchSize);
#else
    auto orb = cv::ORB::create();

    orb->setMaxFeatures(param.nfeature);
    orb->setEdgeThreshold(param.edgeThreshold);
    orb->setNLevels(param.nlevels);
    orb->setFastThreshold(param.fastThreshold);
    orb->setPatchSize(param.patchSize);
    orb->setWTA_K(param.WTA_K);
#endif

    std::vector<cv::KeyPoint> keypoint;
    keypoint.reserve(param.nfeature);
    cv::Mat descriptor;
#if CV_MAJOR_VERSION < 3
    orb->detect(img, keypoint);
    orb->compute(img, keypoint, descriptor);
#else
    orb->detectAndCompute(img, cv::noArray(), keypoint, descriptor, false);
#endif
    return {std::move(keypoint), std::move(descriptor)};
}

std::size_t test_orb_param(const cv::Mat& image_model, const cv::Mat& image_test, const ORB_Param& val)
{
    auto [keypoint_model, descriptor_model] = keypoint_compute(image_model, val);
    auto [keypoint_test, descriptor_test] = keypoint_compute(image_test, val);

#if CV_MAJOR_VERSION < 3
    auto matcher = cv::DescriptorMatcher::create("NORM_HAMMING");
#else
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
#endif

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptor_model, descriptor_test, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> filtered_matches = filter_match(keypoint_model, keypoint_test, knn_matches, ratio_thresh);
    return filtered_matches.size();
}

void keypoint_approach(const cv::Mat& image_model, const cv::Mat& image_test)
{
    cv::namedWindow("pretreat model", cv::WINDOW_NORMAL);
    cv::imshow("pretreat model", image_model);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::namedWindow("pretreat test", cv::WINDOW_NORMAL);
    cv::imshow("pretreat test", image_test);
    cv::waitKey(0);
    cv::destroyAllWindows();

    ORB_Param param_model, test_param;
    param_model.nfeature = 10000;

    test_param.nfeature = 10000;
    // test_param.nlevels = 16;
    test_param.WTA_K = 4;
    test_param.edgeThreshold = 100;

    auto [keypoint_model, descriptor_model] = keypoint_compute(image_model, param_model);
    auto [keypoint_test, descriptor_test] = keypoint_compute(image_test, test_param);

#if CV_MAJOR_VERSION < 3
    auto matcher = cv::DescriptorMatcher::create("NORM_HAMMING");
#else
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
#endif

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptor_model, descriptor_test, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> filtered_matches = filter_match(keypoint_model, keypoint_test, knn_matches, ratio_thresh);

    cv::Mat img_keypoint_model, img_keypoint_test, img_keypoint_matches;

    cv::drawKeypoints(image_model, keypoint_model, img_keypoint_model, cv::Scalar(0, 255, 0));
    cv::drawKeypoints(image_test, keypoint_test, img_keypoint_test, cv::Scalar(0, 255, 0));
    cv::drawMatches(image_model, keypoint_model, image_test, keypoint_test, filtered_matches, img_keypoint_matches);

    cv::namedWindow("Keypoint model", cv::WINDOW_NORMAL);
    cv::imshow("Keypoint model", img_keypoint_model);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::namedWindow("Keypoint test", cv::WINDOW_NORMAL);
    cv::imshow("Keypoint test", img_keypoint_test);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::namedWindow("Keypoint matches", cv::WINDOW_NORMAL);
    cv::imshow("Keypoint matches", img_keypoint_matches);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

std::vector<cv::Vec4i> hough_approach(const cv::Mat& img)
{
    double hysteresis_threshold1 = 0;
    double hysteresis_threshold2 = 0;
    int aperture_size = 5;
    bool gradient = false;
    cv::Mat canny_output;

    cv::Canny(img, canny_output, hysteresis_threshold1, hysteresis_threshold2, aperture_size, gradient);

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

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(canny_output, lines, 1, CV_PI / 180, 100, 0, 0);

    return lines;
}

struct bound
{
    std::size_t ibegin = 0;
    int best_max = 0;
    std::size_t iend = 0;
    int best_min = 0;

    int quality() const { return best_max + -best_min; }
};

bound find_histo_bound(const std::vector<int>& histo)
{
    bound result;
    for(std::size_t i = 0; i < histo.size() - 1; i++)
    {
        int diff = histo[i] - histo[i + 1];
        if(diff < result.best_min)
        {
            result.ibegin = i + 1;
            result.best_min = diff;
        }
        if(diff > result.best_max)
        {
            result.iend = i;
            result.best_max = diff;
        }
    }
    return result;
}

struct line_bound
{
    std::size_t ibegin;
    std::size_t iend;
    std::size_t size;
};

line_bound find_longest_line(const cv::Mat& img)
{
    line_bound result {0, 0, 0};
    std::size_t max = 0;

    std::size_t ibegin = 0;
    std::size_t current = 0;

    const uint8_t* arr = img.ptr<uint8_t>(0, 0);
    std::size_t iend = img.cols;
    for(std::size_t i = 0; i < iend; i++)
    {
        if(arr[i] > 0)
        {
            if(current == 0)
            {
                ibegin = i;
                current++;
            }
            else
            {
                current++;
            }
        }
        else
        {
            if(current > result.size)
            {
                result.size = current;
                result.ibegin = ibegin;
                result.iend = i;
            }
            current = 0;
            ibegin = 0;
        }
    }
    if(current > result.size)
    {
        result.size = current;
        result.ibegin = ibegin;
        result.iend = iend;
    }

    return result;
}

std::vector<std::array<int, 4>> histogram_approach(const cv::Mat& img)
{
    cv::namedWindow("img", cv::WINDOW_NORMAL);
    cv::imshow("img", img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    const int row_size = img.rows;
    std::vector<std::array<int, 4>> result;

    std::vector<line_bound> horz_histo;
    horz_histo.resize(row_size);

    for(int i = 0; i < row_size; i++)
    {
        horz_histo[i] = find_longest_line(img.row(i));
    }

    auto minmax = std::minmax_element(horz_histo.begin(), horz_histo.end(),
        [](const line_bound& lval, const line_bound& rval) { return lval.size < rval.size; });

    auto upper_threshold = minmax.second->size;

    auto upper_b = std::find_if(horz_histo.begin(), horz_histo.end(),
        [upper_threshold](const line_bound& v) { return v.size >= upper_threshold; });

    std::size_t upper_i = std::distance(horz_histo.begin(), upper_b);
    std::size_t lower_i = upper_i + minmax.second->size;
    upper_i *= static_cast<std::size_t>(0.95);
    lower_i *= static_cast<std::size_t>(1.05);
    upper_b->ibegin *= static_cast<std::size_t>(0.95);
    upper_b->iend *= static_cast<std::size_t>(1.05);

    result.push_back({int(upper_b->ibegin), int(upper_i), int(upper_b->iend), int(upper_i)});
    result.push_back({int(upper_b->ibegin), int(upper_i), int(upper_b->ibegin), int(lower_i)});
    result.push_back({int(upper_b->iend), int(upper_i), int(upper_b->iend), int(lower_i)});
    result.push_back({int(upper_b->ibegin), int(lower_i), int(upper_b->iend), int(lower_i)});

    return result;
}

template <typename T> cv::Point center_of_mass(const T& list)
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

auto mean_distance(const cv::Point& p, const std::vector<cv::Point>& list)
{
    double dis = 0;
    for(const auto& v : list)
    {
        dis += euclideanDistance(p, v);
    }
    return dis / list.size();
}

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
    const cv::Mat& pretreat, const cv::Mat& original, bool agregate_contour = true, bool debug = false)
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
        distances[i] = std::abs(mratios - ellipses[i].size.aspectRatio());
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
        cv::ellipse2Poly(rect.center, size, rect.angle, 0, 360, 1, result);
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
            cv::FILLED
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