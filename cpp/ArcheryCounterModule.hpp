#pragma once

#include <array>
#include <string>

struct EllipseData {
    float centerX;
    float centerY;
    float width;   // full bounding-box width  (semi-axis rx = width  / 2)
    float height;  // full bounding-box height (semi-axis ry = height / 2)
    float angle;   // rotation in degrees, same convention as cv::RotatedRect::angle
};

struct ArcheryResult {
    std::array<EllipseData, 10> rings; // index 0 = outermost, index 9 = bullseye
    bool success = false;
    std::string error;
};

/**
 * Load the image at imagePath (filesystem path, no file:// prefix),
 * detect the 10 archery target rings, and return their ellipse parameters.
 */
ArcheryResult processImageFile(const std::string& imagePath);
