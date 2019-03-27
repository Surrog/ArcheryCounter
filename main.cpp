#include <opencv2/opencv.hpp>
#include <stdio.h>

constexpr const char* model = R".(C:\Users\fancel\Documents\ArcheryCounter\Images\model\A53C085.jpg).";
constexpr const char* test = R".(C:\Users\fancel\Documents\ArcheryCounter\Images\20190325_193217.jpg).";

std::pair<std::vector<cv::KeyPoint>, cv::Mat> compute_keypoint(const cv::Mat& image)
{
    auto orb = cv::ORB::create();
    std::vector<cv::KeyPoint> result;
    cv::Mat descriptor;
    orb->detectAndCompute(image, cv::noArray(), result, descriptor);
    return {result, descriptor};
}

int main(int argc, char** argv)
{
    cv::Mat image_model = cv::imread(model, cv::IMREAD_GRAYSCALE);
    cv::Mat image_test = cv::imread(test, cv::IMREAD_GRAYSCALE);

    if(!image_model.data || !image_test.data)
    {
        printf("No image data \n");
        return -1;
    }

    auto [keypoint_model, descriptor_model] = compute_keypoint(image_model);
    auto [keypoint_test, descriptor_test] = compute_keypoint(image_test);

    //auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    //std::vector<std::vector<cv::DMatch>> knn_matches;
    //matcher->knnMatch(descriptor_model, descriptor_test, knn_matches, 2);

    //const float ratio_thresh = 0.75f;
    //std::vector<cv::DMatch> good_matches;
    //for(std::size_t i = 0; i < knn_matches.size(); i++)
    //{
    //    if(knn_matches[i][0].distance < (ratio_thresh * knn_matches[i][1].distance))
    //    {
    //        good_matches.push_back(knn_matches[i][0]);
    //    }
    //}

    cv::Mat img_matches;

    cv::Mat detected;
    cv::drawKeypoints(image_model, keypoint_model, detected, cv::Scalar(0, 255, 0));

    cv::namedWindow("Display Image", cv::WINDOW_NORMAL);
    cv::imshow("Display Image", detected);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}