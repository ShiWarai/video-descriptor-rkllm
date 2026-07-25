#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace vlm {

class ImagePreprocess {
public:
    static cv::Mat expandToSquare(const cv::Mat& img,
                                  const cv::Scalar& background = cv::Scalar(127, 127, 127));

    static cv::Mat resizeToModel(const cv::Mat& img, int width, int height);

    static cv::Mat prepare(const cv::Mat& bgr, int model_width, int model_height);
};

}  // namespace vlm
