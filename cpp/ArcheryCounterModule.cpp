// Platform-independent C++ entry point for the React Native native module.
// Both the iOS Obj-C++ bridge and the Android JNI bridge call processImageFile().
// Note: debug=false is hardcoded — cv::imshow/cv::waitKey are not available on mobile.

#include "ArcheryCounterModule.hpp"
#include "ellispes_approach.hpp"

#include <opencv2/opencv.hpp>
#include <stdexcept>

ArcheryResult processImageFile(const std::string& imagePath)
{
    ArcheryResult out;
    try
    {
        // imread returns a BGR image, which is what ellipses::find_target expects
        // (colour_filters.cpp uses cv::COLOR_RGB2HSV_FULL, consistent with BGR input
        // from imread — matching the behaviour of the original main.cpp).
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty())
        {
            out.error = "Could not load image: " + imagePath;
            return out;
        }

        std::array<cv::RotatedRect, 10> rings = ellipses::find_target(image, false);

        for (int i = 0; i < 10; i++)
        {
            out.rings[i] = {
                rings[i].center.x,
                rings[i].center.y,
                rings[i].size.width,
                rings[i].size.height,
                rings[i].angle,
            };
        }

        out.success = true;
    }
    catch (const cv::Exception& e)
    {
        out.error = std::string("OpenCV error: ") + e.what();
    }
    catch (const std::exception& e)
    {
        out.error = e.what();
    }
    return out;
}
