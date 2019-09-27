#include "keypoint_approach.hpp"

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
