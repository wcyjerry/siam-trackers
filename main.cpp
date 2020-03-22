#include <iostream>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "Tracker.h"

const cv::Scalar COLOR_GREEN(0, 255, 0);

int main() {
    _putenv_s("OMP_NUM_THREADS", "8");

    std::vector<TorchModule> backbone;
    torch::Tensor z_crop = torch::zeros({ 1, 3, Tracker::EXEMPLAR_SIZE, Tracker::EXEMPLAR_SIZE }).cuda();
    torch::Tensor x_crop = torch::zeros({ 1, 3, Tracker::INSTANCE_SIZE, Tracker::INSTANCE_SIZE }).cuda();
    torch::List<torch::Tensor> pre_zf;
    torch::List<torch::Tensor> pre_xf;
    int nextUsedLayerIdx = 0;
    for (int i = 0; i < 8; i++) {
        TorchModule layer = torch::jit::load("siamrpnpp_backbone_mobilev2_layer" + std::to_string(i) + ".pt", torch::kCUDA);
        layer.eval();
        z_crop = layer.forward({ z_crop }).toTensor();
        x_crop = layer.forward({ x_crop }).toTensor();
        if (i == Tracker::BACKBONE_USED_LAYERS[nextUsedLayerIdx]) {
            pre_zf.push_back(z_crop);
            pre_xf.push_back(x_crop);
            nextUsedLayerIdx++;
        }
        backbone.push_back(layer);
    }

    TorchModule neck = torch::jit::load("siamrpnpp_neck_adjust_all_layer.pt", torch::kCUDA);
    neck.eval();
    torch::List<torch::Tensor> zf, xf;
    zf = neck.forward({ pre_zf }).toTensorList();
    xf = neck.forward({ pre_xf }).toTensorList();

    std::vector<TorchModule> rpns;
    for (int i = 0; i < 3; i++) {
        TorchModule rpn = torch::jit::load("siamrpnpp_rpn_head_multi_rpn" + std::to_string(i + 2) + ".pt", torch::kCUDA);
        rpn.eval();
        rpn.forward({ zf.get(i), xf.get(i) });
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
    getchar();
    return 0;
}