#pragma once

#include <torch/script.h>
#include <opencv2/opencv.hpp>

typedef torch::jit::script::Module TorchModule;

class Tracker {
    TorchModule neck;
    std::vector<TorchModule> backbone, rpns;

    cv::Rect bounding_box;

    bool ready_to_track = false;
    std::string obj_id;

    // TODO: What are these?
    cv::Scalar channel_average;
    torch::List<torch::Tensor> zf;
    torch::Tensor rpn_weights = torch::softmax(torch::tensor(torch::ArrayRef<float>({ 1, 1, 1 })), 0);
    torch::Tensor anchors;
    torch::Tensor window;

    torch::List<torch::Tensor> backbone_forward(torch::Tensor crop);

    // TODO: What are these?
    void load_networks_instantly();
    void generate_anchors();
    int calculate_s_z();
    torch::Tensor get_subwindow(cv::Mat frame, int exampler_size, int original_size);
    torch::Tensor convert_score(torch::Tensor cls);
    torch::Tensor convert_bbox(torch::Tensor loc);

public:
    // TODO: What are these?
    static const int BACKBONE_USED_LAYERS_NUM = 3;
    static const int BACKBONE_USED_LAYERS[BACKBONE_USED_LAYERS_NUM];
    static const float CONTEXT_AMOUNT;
    static const int EXEMPLAR_SIZE = 127;
    static const int INSTANCE_SIZE = 255;
    static const int ANCHOR_STRIDE = 8;
    static const int ANCHOR_RATIOS_NUM = 5;
    static const float ANCHOR_RATIOS[ANCHOR_RATIOS_NUM];
    static const int ANCHOR_SCALES_NUM = 1;
    static const float ANCHOR_SCALES[ANCHOR_SCALES_NUM];
    static const float TRACK_PENALTY_K;
    static const float TRACK_WINDOW_INFLUENCE;
    static const float TRACK_LR;
    static const int TRACK_BASE_SIZE = 8;

    Tracker(std::vector<TorchModule> backbone, TorchModule neck, std::vector<TorchModule> rpns) : backbone(backbone), neck(neck), rpns(rpns) {
        generate_anchors();
        load_networks_instantly();
    };

    void init(cv::Mat frame, cv::Rect roi, std::string id);
    cv::Rect track(cv::Mat frame);

    bool is_ready_to_track() {
        return ready_to_track;
    }

    void stop_tracking() {
        // TODO: More proper cleanups
        ready_to_track = false;
    }

    std::string get_obj_id() {
        return obj_id;
    }
};
