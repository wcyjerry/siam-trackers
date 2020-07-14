#include "Tracker.h"

class TrackerSiamRPNPP : public Tracker {
	std::vector<TorchModule> backbone, rpns;
	torch::List<torch::Tensor> backbone_forward(torch::Tensor crop);

public:
	TrackerSiamRPNPP(std::vector<TorchModule> backbone, TorchModule neck, std::vector<TorchModule> rpns) : backbone(backbone), rpns(rpns), Tracker(neck) {}
	void load_networks_instantly();
	cv::Rect track(cv::Mat frame);
};