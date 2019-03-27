#include <opencv2/opencv.hpp>
#include <stdio.h>

constexpr const char* model = R".(C:\Users\fancel\Documents\ArcheryCounter\Images\model\A53C085.jpg).";
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

void keypoint_approach(const cv::Mat& image_model, const cv::Mat& image_test) {
    auto orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoint_model, keypoint_test;
    cv::Mat descriptor_model, descriptor_test;

    orb->detectAndCompute(image_model, cv::noArray(), keypoint_model, descriptor_model, false);
    orb->detectAndCompute(image_test, cv::noArray(), keypoint_test, descriptor_test, false);

    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptor_model, descriptor_test, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> filtered_matches = filter_match(keypoint_model, keypoint_test, knn_matches, ratio_thresh);

    cv::Mat img_matches;

    // cv::drawKeypoints(image_model, keypoint_model, detected, cv::Scalar(0, 255, 0));
    drawMatches(image_model, keypoint_model, image_test, keypoint_test, filtered_matches, img_matches);

    cv::namedWindow("Display Image", cv::WINDOW_NORMAL);
    cv::imshow("Display Image", img_matches);
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

	//keypoint_approach(image_model, image_test);
	return 0;
}