#pragma once

#include "Tracker.h"

struct siam_mask_model {
	TorchModule backbone_conv;
	TorchModule backbone_bn;
	std::vector<TorchModule> backbone_layers;
	TorchModule neck;
	TorchModule rpn_head;
	TorchModule mask_head;
	TorchModule refine_head;
};

class TrackerSiamMask : public Tracker {
	static const int BACKBONE_USED_LAYERS_NUM = 4;
	static const int BACKBONE_USED_LAYERS[BACKBONE_USED_LAYERS_NUM];
	static const int MASK_OUTPUT_SIZE = 127;
	static const float MASK_THRESHOLD;

	siam_mask_model model;
	torch::List<torch::Tensor> backbone_forward(torch::Tensor input);
	torch::List<torch::Tensor> neck_forward(torch::List<torch::Tensor> input);

public:
	TrackerSiamMask(siam_mask_model model) : model(model) {
		TRACK_PENALTY_K = 0.1;
		TRACK_WINDOW_INFLUENCE = 0.41;
		TRACK_LR = 0.32;
	}

	void load_networks_instantly();
	virtual track_result track(cv::Mat frame);
};
