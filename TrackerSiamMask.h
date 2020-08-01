#pragma once

#include "Tracker.h"

class TrackerSiamMask : public Tracker {
	static const int BACKBONE_USED_LAYERS_NUM = 4;
	static const int BACKBONE_USED_LAYERS[BACKBONE_USED_LAYERS_NUM];
	static const int MASK_OUTPUT_SIZE = 127;
	static const float MASK_THRESHOLD;

	TorchModule backbone_conv, backbone_bn;
	std::vector<TorchModule> backbone_layers;
	TorchModule rpn_head, mask_head, refine_head;
	torch::List<torch::Tensor> backbone_forward(torch::Tensor crop);

public:
	TrackerSiamMask(TorchModule backbone_conv, TorchModule backbone_bn, std::vector<TorchModule> backbone_layers, TorchModule neck, TorchModule rpn_head, TorchModule mask_head, TorchModule refine_head)
		: backbone_conv(backbone_conv), backbone_bn(backbone_bn), backbone_layers(backbone_layers), rpn_head(rpn_head), mask_head(mask_head), refine_head(refine_head), Tracker(neck) {
		TRACK_PENALTY_K = 0.1;
		TRACK_WINDOW_INFLUENCE = 0.41;
		TRACK_LR = 0.32;
		is_mask = true;
	}

	void load_networks_instantly();
	virtual track_result track(cv::Mat frame);
};
