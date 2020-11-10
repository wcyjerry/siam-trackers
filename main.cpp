#include <iostream>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "TrackerSiamRPNPP.h"
#include "TrackerSiamMask.h"

#ifdef _WIN32
#include "windows.h"
#endif

const cv::Scalar COLOR_BLUE(255, 0, 0);

int main(int argc, char **argv) {
    #ifdef _WIN32
    _putenv_s("OMP_NUM_THREADS", "8");
    LoadLibraryA("torch_cuda.dll");
    #endif

    // TODO: Should we `eval` these modules?
    // SiamMask
    siam_mask_model mask_model;
    mask_model.backbone_conv = torch::jit::load("siammask_backbone_resnet_conv1.pt", torch::kCUDA);
    mask_model.backbone_bn = torch::jit::load("siammask_backbone_resnet_bn1.pt", torch::kCUDA);
    for (int i = 1; i < 9; i++) {
        TorchModule layer = torch::jit::load("siammask_backbone_resnet_layer" + std::to_string(i) + ".pt", torch::kCUDA);
        mask_model.backbone_layers.push_back(layer);
    }
    mask_model.neck = torch::jit::load("siammask_neck_adjust_all_layer.pt", torch::kCUDA);
    mask_model.rpn_head = torch::jit::load("siammask_rpn_head.pt", torch::kCUDA);
    mask_model.mask_head = torch::jit::load("siammask_mask_head.pt", torch::kCUDA);
    mask_model.refine_head = torch::jit::load("siammask_refine_head.pt", torch::kCUDA);
    TrackerSiamMask tracker_siam_mask(mask_model);
    tracker_siam_mask.load_networks_instantly();

    // SiamRPN++
    siam_rpnpp_model rpnpp_model;
    for (int i = 0; i < 8; i++) {
        TorchModule layer = torch::jit::load("siamrpnpp_backbone_mobilev2_layer" + std::to_string(i) + ".pt", torch::kCUDA);
        rpnpp_model.backbone.push_back(layer);
    }
    rpnpp_model.neck = torch::jit::load("siamrpnpp_neck_adjust_all_layer.pt", torch::kCUDA);
    for (int i = 0; i < 3; i++) {
        TorchModule rpn = torch::jit::load("siamrpnpp_rpn_head_multi_rpn" + std::to_string(i + 2) + ".pt", torch::kCUDA);
        rpnpp_model.rpns.push_back(rpn);
    }
    TrackerSiamRPNPP tracker_siam_rpnpp(rpnpp_model);
    tracker_siam_rpnpp.load_networks_instantly();

    // Start with SiamMask
    Tracker* tracker = &tracker_siam_mask;

    // Load video
    std::string src = argc > 1 ? argv[1] : "bag.avi";
    cv::VideoCapture video_cap(src, cv::CAP_FFMPEG);

    // Start processing
    cv::Mat frame_original;
    cv::Rect roi;
    int frame_count = 0;
    std::chrono::steady_clock::time_point time_start;
    while (video_cap.read(frame_original)) {
        if (frame_count == 0) {
            roi = cv::selectROI(src, frame_original, false);
            tracker->init(frame_original, roi);
            frame_count = 1;
            time_start = std::chrono::steady_clock::now();
        }
        else {
            frame_count++;
            track_result res = tracker->track(frame_original);

            cv::Mat frame_show = frame_original.clone();

            cv::Point2f vertices[4];
            res.bbox.points(vertices);
            for (int i = 0; i < 4; i++) {
                line(frame_show, vertices[i], vertices[(i + 1) % 4], COLOR_BLUE, 2);
            }

            if (!res.mask.empty()) {
                std::vector<cv::Mat> channels{ res.mask * COLOR_BLUE[0], res.mask * COLOR_BLUE[1], res.mask * COLOR_BLUE[2] };
                cv::Mat colored_mask;
                cv::merge(channels, colored_mask);
                cv::addWeighted(frame_show, 0.77, colored_mask, 0.23, -1, frame_show);
                cv::drawContours(frame_show, res.contours, -1, COLOR_BLUE, 2);
            }

            cv::putText(
                frame_show,
                std::to_string(frame_count / (std::chrono::duration<double>(std::chrono::steady_clock::now() - time_start)).count()) + " FPS",
                cv::Point(20, 20),
                cv::FONT_HERSHEY_COMPLEX_SMALL,
                1.f,
                COLOR_BLUE
            );
            cv::imshow(src, frame_show);

            char k = cv::waitKey(1);
            if (k == 'm') {
                cv::Rect prev_bbox = tracker->get_bounding_box();
                tracker->stop_tracking();
                if (tracker == &tracker_siam_mask) {
                    tracker = &tracker_siam_rpnpp;
                }
                else {
                    tracker = &tracker_siam_mask;
                }
                tracker->init(frame_original, prev_bbox);
            }
        }
    }
    return 0;
}
