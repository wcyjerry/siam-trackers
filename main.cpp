#include <iostream>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "Tracker.h"

const cv::Scalar COLOR_GREEN(0, 255, 0);

int main() {
    _putenv_s("OMP_NUM_THREADS", "8");

    // TODO: Should we `eval` these modules?
    std::vector<TorchModule> backbone;
    for (int i = 0; i < 8; i++) {
        TorchModule layer = torch::jit::load("siamrpnpp_backbone_mobilev2_layer" + std::to_string(i) + ".pt", torch::kCUDA);
        backbone.push_back(layer);
    }
    TorchModule neck = torch::jit::load("siamrpnpp_neck_adjust_all_layer.pt", torch::kCUDA);
    std::vector<TorchModule> rpns;
    for (int i = 0; i < 3; i++) {
        TorchModule rpn = torch::jit::load("siamrpnpp_rpn_head_multi_rpn" + std::to_string(i + 2) + ".pt", torch::kCUDA);
        rpns.push_back(rpn);
    }
    Tracker tracker(backbone, neck, rpns);

    std::string src = "bag.avi";
    cv::VideoCapture video_cap(src);

    cv::Mat frame;
    cv::Rect roi;
    int frame_count = 0;
    std::chrono::steady_clock::time_point time_start;
    while (video_cap.read(frame)) {
        if (frame_count == 0) {
            roi = cv::selectROI(src, frame, false);
            tracker.init(frame, roi);
            frame_count = 1;
            time_start = std::chrono::steady_clock::now();
        }
        else {
            frame_count++;
            cv::Rect bbox = tracker.track(frame);
            cv::rectangle(frame, bbox, COLOR_GREEN, 3);
            cv::putText(
                frame,
                std::to_string(frame_count / (std::chrono::duration<double>(std::chrono::steady_clock::now() - time_start)).count()) + " FPS",
                cv::Point(20, 20),
                cv::FONT_HERSHEY_COMPLEX_SMALL,
                1.f,
                COLOR_GREEN
            );
            cv::imshow(src, frame);
            cv::waitKey(1);
        }
    }
    return 0;
}