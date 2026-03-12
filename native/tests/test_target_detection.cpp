#include "ellispes_approach.hpp"
#include "sources.hpp"

#include <opencv2/opencv.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ── helpers ──────────────────────────────────────────────────────────────────

struct Result
{
    std::string image;
    bool passed = false;
    std::string reason;
};

static Result test_image(const fs::path& path)
{
    Result r;
    r.image = path.filename().string();

    cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty())
    {
        r.reason = "could not load image";
        return r;
    }

    std::array<cv::RotatedRect, 10> rings;
    try
    {
        rings = ellipses::find_target(image, false);
    }
    catch (const std::exception& e)
    {
        r.reason = std::string("find_target threw: ") + e.what();
        return r;
    }

    // All 10 rings must have positive size.
    for (int i = 0; i < 10; i++)
    {
        if (rings[i].size.width <= 0.f || rings[i].size.height <= 0.f)
        {
            r.reason = "ring " + std::to_string(i) + " has zero/negative size";
            return r;
        }
    }

    // All ring centers must lie within the image.
    for (int i = 0; i < 10; i++)
    {
        const auto& c = rings[i].center;
        if (c.x < 0.f || c.x > float(image.cols) || c.y < 0.f || c.y > float(image.rows))
        {
            r.reason = "ring " + std::to_string(i) + " center (" + std::to_string(c.x) + ", "
                     + std::to_string(c.y) + ") is outside image (" + std::to_string(image.cols) + "x"
                     + std::to_string(image.rows) + ")";
            return r;
        }
    }

    // All rings must be roughly concentric.
    // The 3 detected rings come from different HSV colour filters (not size-ordered),
    // and linear interpolation can drift the extrapolated centers; 100 px is a safe bound.
    const cv::Point2f origin = rings[0].center;
    for (int i = 1; i < 10; i++)
    {
        const float dx = rings[i].center.x - origin.x;
        const float dy = rings[i].center.y - origin.y;
        const float dist = std::sqrt(dx * dx + dy * dy);
        if (dist > 100.f)
        {
            r.reason = "ring " + std::to_string(i) + " center is " + std::to_string(dist)
                     + " px from ring 0 (max 100)";
            return r;
        }
    }

    r.passed = true;
    r.reason = "10 concentric rings detected";
    return r;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main()
{
    const fs::path images_dir(IMAGE_TEST_DIR);

    if (!fs::is_directory(images_dir))
    {
        std::cerr << "ERROR: image directory not found: " << images_dir << "\n";
        return 1;
    }

    // Collect .jpg files directly under IMAGE_TEST_DIR (skip the model/ subdirectory).
    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(images_dir))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg")
            images.push_back(entry.path());
    }

    if (images.empty())
    {
        std::cerr << "ERROR: no .jpg files found in " << images_dir << "\n";
        return 1;
    }

    std::sort(images.begin(), images.end());

    int passed = 0, failed = 0;
    for (const auto& img : images)
    {
        Result r = test_image(img);
        const char* tag = r.passed ? "PASS" : "FAIL";
        std::cout << "[" << tag << "] " << r.image << " — " << r.reason << "\n";
        r.passed ? ++passed : ++failed;
    }

    std::cout << "\n" << passed << "/" << (passed + failed) << " images passed.\n";
    return failed > 0 ? 1 : 0;
}
