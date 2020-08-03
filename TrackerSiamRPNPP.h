#pragma once

#include "Tracker.h"

struct siam_rpnpp_model {
	std::vector<TorchModule> backbone;
	TorchModule neck;
	std::vector<TorchModule> rpns;
};

class TrackerSiamRPNPP : public Tracker {
	static const int BACKBONE_USED_LAYERS_NUM = 3;
	static const int BACKBONE_USED_LAYERS[BACKBONE_USED_LAYERS_NUM];

	siam_rpnpp_model model;
	torch::List<torch::Tensor> backbone_forward(torch::Tensor crop);
	torch::List<torch::Tensor> neck_forward(torch::List<torch::Tensor> input);

public:
	TrackerSiamRPNPP(siam_rpnpp_model model) : model(model) {
		TRACK_PENALTY_K = 0.04;
		TRACK_WINDOW_INFLUENCE = 0.4;
		TRACK_LR = 0.5;
	}

	void load_networks_instantly();
	virtual track_result track(cv::Mat frame);
};
