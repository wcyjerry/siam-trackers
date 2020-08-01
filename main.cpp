#include <iostream>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "TrackerSiamRPNPP.h"
#include "TrackerSiamMask.h"

const cv::Scalar COLOR_GREEN(0, 255, 0);

int main(int argc, char **argv) {
    _putenv_s("OMP_NUM_THREADS", "8");

    // TODO: Should we `eval` these modules?
    /*std::vector<TorchModule> backbone;
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
    TrackerSiamRPNPP tracker(backbone, neck, rpns);*/

    TorchModule backbone_conv = torch::jit::load("siammask_backbone_resnet_conv1.pt", torch::kCUDA);
    TorchModule backbone_bn = torch::jit::load("siammask_backbone_resnet_bn1.pt", torch::kCUDA);
    std::vector<TorchModule> backbone_layers;
    for (int i = 1; i < 9; i++) {
        TorchModule layer = torch::jit::load("siammask_backbone_resnet_layer" + std::to_string(i) + ".pt", torch::kCUDA);
        backbone_layers.push_back(layer);
    }
    TorchModule neck = torch::jit::load("siammask_neck_adjust_all_layer.pt", torch::kCUDA);
    TorchModule rpn_head = torch::jit::load("siammask_rpn_head.pt", torch::kCUDA);
    TorchModule mask_head = torch::jit::load("siammask_mask_head.pt", torch::kCUDA);
    TorchModule refine_head = torch::jit::load("siammask_refine_head.pt", torch::kCUDA);
    TrackerSiamMask tracker(backbone_conv, backbone_bn, backbone_layers, neck, rpn_head, mask_head, refine_head);

    tracker.load_networks_instantly();

    std::string src = argc > 1 ? argv[1] : "bag.avi";
    cv::VideoCapture video_cap(src, cv::CAP_FFMPEG);

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
            track_result res = tracker.track(frame);

            cv::Point2f vertices[4];
            res.bbox.points(vertices);
            for (int i = 0; i < 4; i++) {
                line(frame, vertices[i], vertices[(i + 1) % 4], COLOR_GREEN, 3);
            }

            if (!res.mask.empty()) {
                cv::addWeighted(frame, 0.77, res.mask, 0.23, -1, frame);
            }

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
