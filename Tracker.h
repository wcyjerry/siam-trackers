#pragma once

#include <torch/script.h>
#include <opencv2/opencv.hpp>

typedef torch::jit::script::Module TorchModule;

class Tracker {

    bool ready_to_track = false;
    std::string obj_id;
    int obj_class_id;
    std::string obj_class_name;

    virtual torch::List<torch::Tensor> backbone_forward(torch::Tensor crop) = 0;

protected:
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

    TorchModule neck;
    cv::Rect bounding_box;

    Tracker(TorchModule neck) : neck(neck) {
        generate_anchors();
    };

    // TODO: What are these
    torch::Tensor change(torch::Tensor r);
    torch::Tensor sz(torch::Tensor w, torch::Tensor h);
    torch::Tensor hann_window(int window_length);

    // TODO: What are these?
    cv::Scalar channel_average;
    torch::List<torch::Tensor> zf;
    torch::Tensor anchors;
    torch::Tensor window;

    // TODO: What are these?
    void generate_anchors();
    int calculate_s_z();
    torch::Tensor get_subwindow(cv::Mat frame, int exampler_size, int original_size);
    torch::Tensor convert_score(torch::Tensor cls);
    torch::Tensor convert_bbox(torch::Tensor loc);

public:

    virtual void init(cv::Mat frame, cv::Rect roi, std::string obj_id = "", int obj_class_id = -1, std::string obj_class_name = "");
    // TODO: https://gitlab.kikaitech.io/kikai-ai/siam-trackers/issues/11
    virtual void load_networks_instantly() = 0;
    virtual cv::Rect track(cv::Mat frame) = 0;

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

    int get_obj_class_id() {
        return obj_class_id;
    }

    std::string get_obj_class_name() {
        return obj_class_name;
    }
};
