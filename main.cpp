#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdio.h>

constexpr const char* model = R".(C:\Users\fancel\Documents\ArcheryCounter\Images\model\cleaned.jpg).";
constexpr const char* test = R".(C:\Users\fancel\Documents\ArcheryCounter\Images\20190325_193217.jpg).";

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
    auto descriptor_type = cv::NORM_HAMMING;
#else
    auto descriptor_type = cv::DescriptorMatcher::BRUTEFORCE_HAMMING;
#endif 

	auto matcher = cv::DescriptorMatcher::create(descriptor_type);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptor_model, descriptor_test, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> filtered_matches = filter_match(keypoint_model, keypoint_test, knn_matches, ratio_thresh);
    return filtered_matches.size();
}

void keypoint_approach(const cv::Mat& image_model, const cv::Mat& image_test)
{
    ORB_Param param_model, test_param;
    param_model.nfeature = 10000;

    test_param.nfeature = 10000;
    // test_param.nlevels = 16;
    test_param.WTA_K = 4;
    test_param.edgeThreshold = 100;

    auto [keypoint_model, descriptor_model] = keypoint_compute(image_model, param_model);
    auto [keypoint_test, descriptor_test] = keypoint_compute(image_test, test_param);

    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptor_model, descriptor_test, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> filtered_matches = filter_match(keypoint_model, keypoint_test, knn_matches, ratio_thresh);

    cv::Mat img_matches;

    // cv::drawKeypoints(image_model, keypoint_model, img_matches, cv::Scalar(0, 255, 0));
    cv::drawKeypoints(image_test, keypoint_test, img_matches, cv::Scalar(0, 255, 0));
    // drawMatches(image_model, keypoint_model, image_test, keypoint_test, filtered_matches, img_matches);

    cv::namedWindow("Display Image", cv::WINDOW_NORMAL);
    cv::imshow("Display Image", img_matches);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void hough_approach(const cv::Mat& img)
{
    cv::Mat pretreat = img.clone();
    cv::cvtColor(pretreat, pretreat, cv::COLOR_RGB2HSV_FULL);
    cv::inRange(pretreat, cv::Scalar(0, 0, 170), cv::Scalar(220, 25, 255), pretreat);

    // cv::GaussianBlur(pretreat, pretreat, cv::Size(15, 15), 1.5, 1.5);
    // cv::erode(pretreat, pretreat, cv::Mat());
    // cv::dilate(pretreat, pretreat, cv::Mat());

    // cv::namedWindow("pretreat", cv::WINDOW_NORMAL);
    // cv::imshow("pretreat", pretreat);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    double hysteresis_threshold1 = 0;
    double hysteresis_threshold2 = 25;
    int aperture_size = 5;
    bool gradient = false;
    cv::Mat canny_output;

    cv::Canny(pretreat, canny_output, hysteresis_threshold1, hysteresis_threshold2, aperture_size, gradient);

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

    cv::Mat display = img.clone();
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(canny_output, lines, 1, CV_PI / 180, 100, 0, 0);

    for(const auto& v : lines)
    {
        cv::line(display, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), cv::Scalar(0, 0, 255), 1, 8);
    }

    cv::namedWindow("HoughLinesP", cv::WINDOW_NORMAL);
    cv::imshow("HoughLinesP", display);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char** argv)
{
    cv::Mat image_model = cv::imread(model, cv::IMREAD_COLOR);
    cv::Mat image_test = cv::imread(test, cv::IMREAD_COLOR);

    if(!image_model.data || !image_test.data)
    {
        printf("No image data \n");
        return -1;
    }

    cv::copyMakeBorder(image_model, image_model, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255));
    // cv::threshold(image_model, image_model, 255 / 2, 255, cv::THRESH_BINARY);
    // cv::threshold(image_test, image_test, 255 / 2, 255, cv::THRESH_BINARY);

    // keypoint_approach(image_model, image_test);
    hough_approach(image_test);
    return 0;
}