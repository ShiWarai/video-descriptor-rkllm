#include "core/image_preprocess.hpp"

namespace vlm {

cv::Mat ImagePreprocess::expandToSquare(const cv::Mat& img, const cv::Scalar& background)
{
    const int width = img.cols;
    const int height = img.rows;
    if (width == height) {
        return img.clone();
    }

    const int size = std::max(width, height);
    cv::Mat result(size, size, img.type(), background);
    const int x_offset = (size - width) / 2;
    const int y_offset = (size - height) / 2;
    img.copyTo(result(cv::Rect(x_offset, y_offset, width, height)));
    return result;
}

cv::Mat ImagePreprocess::resizeToModel(const cv::Mat& img, int width, int height)
{
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    return resized;
}

cv::Mat ImagePreprocess::prepare(const cv::Mat& bgr, int model_width, int model_height)
{
    return resizeToModel(expandToSquare(bgr), model_width, model_height);
}

}  // namespace vlm
