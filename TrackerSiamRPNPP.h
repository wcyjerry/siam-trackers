#pragma once

#include "Tracker.h"

class TrackerSiamRPNPP : public Tracker {
	static const int BACKBONE_USED_LAYERS_NUM = 3;
	static const int BACKBONE_USED_LAYERS[BACKBONE_USED_LAYERS_NUM];

	std::vector<TorchModule> backbone, rpns;
	torch::List<torch::Tensor> backbone_forward(torch::Tensor crop);

public:
	TrackerSiamRPNPP(std::vector<TorchModule> backbone, TorchModule neck, std::vector<TorchModule> rpns)
		: backbone(backbone), rpns(rpns), Tracker(neck) {
		TRACK_PENALTY_K = 0.04;
		TRACK_WINDOW_INFLUENCE = 0.4;
		TRACK_LR = 0.5;
	}

	void load_networks_instantly();
	cv::Rect track(cv::Mat frame);
};
