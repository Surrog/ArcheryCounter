#include "sources.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdio.h>

constexpr const char* model = IMAGE_TEST_DIR R".(\model\cleaned.jpg).";
constexpr const char* test = IMAGE_TEST_DIR R".(\20190325_195801.jpg).";

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
    cv::inRange(pretreat, cv::Scalar(132, 64, 0), cv::Scalar(142, 255, 255), pretreat);
    return pretreat;
}

cv::Mat filter_red(const cv::Mat& img)
{
    cv::Mat pretreat = img.clone();
    cv::cvtColor(pretreat, pretreat, cv::COLOR_RGB2HSV_FULL);
    cv::inRange(pretreat, cv::Scalar(164, 64, 0), cv::Scalar(172, 255, 255), pretreat);
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
    // cv::namedWindow("pretreat", cv::WINDOW_NORMAL);
    // cv::imshow("pretreat", img);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

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
    upper_i *= 0.95;
    lower_i *= 1.05;
    upper_b->ibegin *= 0.95;
    upper_b->iend *= 1.05;

    result.push_back({int(upper_b->ibegin), int(upper_i), int(upper_b->iend), int(upper_i)});
    result.push_back({int(upper_b->ibegin), int(upper_i), int(upper_b->ibegin), int(lower_i)});
    result.push_back({int(upper_b->iend), int(upper_i), int(upper_b->iend), int(lower_i)});
    result.push_back({int(upper_b->ibegin), int(lower_i), int(upper_b->iend), int(lower_i)});

    return result;
}

cv::Point center_of_mass(std::vector<cv::Point>& list)
{
    double mean_x = 0;
    double mean_y = 0;
    for(auto& p : list)
    {
        mean_x += double(p.x) / list.size();
        mean_y += double(p.y) / list.size();
    }
    return cv::Point(int(mean_x), int(mean_y));
}

std::vector<cv::Point>
cleanup_center_points(std::vector<cv::Point> points, const cv::Mat& img)
{
    std::ptrdiff_t erase_count = 0;
    cv::RotatedRect ellipse;
    do
    {
        ellipse = cv::fitEllipse(points);

        auto tmp = img.clone();
        cv::drawContours(tmp, std::vector<std::vector<cv::Point>> {points}, 0, cv::Scalar(255, 0, 0), 3);
        cv::ellipse(tmp, ellipse, cv::Scalar(0, 255, 0), 3);
        cv::namedWindow("Cleanup", cv::WINDOW_NORMAL);
        cv::imshow("Cleanup", tmp);
        cv::waitKey(0);
        cv::destroyAllWindows();

        auto lower_threshold = ellipse.size.width * 0.45;
        auto upper_threshold = ellipse.size.width * .59;

		cv::Point center = ellipse.center;

        auto it = std::remove_if(points.begin(), points.end(),
            [center, lower_threshold, upper_threshold](const cv::Point& p) {
                auto dis = euclideanDistance(p, center);
                return dis < lower_threshold || dis > upper_threshold;
            });
        erase_count = std::distance(it, points.end());
        points.erase(it, points.end());
    } while(erase_count > 0 && points.size());
    return points;
}

std::pair<std::vector<cv::Point>, cv::RotatedRect> detect_circle_approach(
    const cv::Mat& pretreat, const cv::Mat& original)
{
    cv::namedWindow("gray", cv::WINDOW_NORMAL);
    cv::imshow("gray", pretreat);
    cv::waitKey(0);

    // std::vector<cv::KeyPoint> result;
    // cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 2, 150, 180, 100);

    std::vector<std::vector<cv::Point>> contours;
    findContours(pretreat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> result;

    for(auto& v : contours)
    {
        if(v.size() > result.size())
        {
            if(std::all_of(v.begin(), v.end(), [](const cv::Point& v) { return v.x > -1 && v.y > -1; }))
            {
                result = std::move(v);
            }
        }
    }

    auto kernel_center = center_of_mass(result);
    auto threshold = kernel_center.x * kernel_center.y / 8000;

    for(auto& v : contours)
    {
        if(v.size() > 5 && euclideanDistance(v[0], kernel_center) < threshold)
        {
            result.insert(result.end(), v.begin(), v.end());
        }
    }

    result = cleanup_center_points(std::move(result), original);

	cv::RotatedRect rect;
    if(result.size())
        rect = cv::fitEllipse(result);

    return {result, rect};
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

    // cv::copyMakeBorder(image_model, image_model, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));
    // cv::threshold(image_model, image_model, 255 / 2, 255, cv::THRESH_BINARY);
    // cv::threshold(image_test, image_test, 255 / 2, 255, cv::THRESH_BINARY);

    /*keypoint_approach(filter_white(pretreatment(image_model)), filter_white(pretreatment(image_test)));
    auto result = hough_approach(filter_white(pretreatment(image_test)));
    */
    // auto result = filter_yellow(pretreatment(image_test));
    // cv::namedWindow("pretreat", cv::WINDOW_NORMAL);
    // cv::imshow("pretreat", result);
    // cv::waitKey(0);

    /*auto result = histogram_approach(filter_white(pretreatment(image_test)));
    for(const auto& v : result)
    {
        cv::line(image_test, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), cv::Scalar(0, 0, 255), 5, 8);
    }
*/
    //auto [contours, ellipse] = detect_circle_approach(filter_yellow(pretreatment(image_test)), image_test);
    //auto [contours, ellipse] = detect_circle_approach(filter_red(pretreatment(image_test)), image_test);
    //auto [contours, ellipse] = detect_circle_approach(filter_blue(pretreatment(image_test)), image_test);
    auto [contours, ellipse] = detect_circle_approach(filter_black(pretreatment(image_test)), image_test);

    cv::drawContours(
        image_test, std::vector<std::vector<cv::Point>> {std::move(contours)}, 0, cv::Scalar(255, 0, 0), 3);
    cv::ellipse(image_test, ellipse, cv::Scalar(0, 255, 0), 3);

    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::imshow("Output", image_test);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}