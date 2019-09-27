#include "histo_approach.hpp"

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
