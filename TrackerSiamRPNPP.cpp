#include "TrackerSiamRPNPP.h"

const float TrackerSiamRPNPP::CONTEXT_AMOUNT = 0.5f;
const int TrackerSiamRPNPP::BACKBONE_USED_LAYERS[BACKBONE_USED_LAYERS_NUM] = { 3, 5, 7 };
const float TrackerSiamRPNPP::ANCHOR_RATIOS[ANCHOR_RATIOS_NUM] = { 0.33, 0.5, 1, 2, 3 };
const float TrackerSiamRPNPP::ANCHOR_SCALES[ANCHOR_SCALES_NUM] = { 8 };
const float TrackerSiamRPNPP::TRACK_PENALTY_K = 0.04;
const float TrackerSiamRPNPP::TRACK_WINDOW_INFLUENCE = 0.4;
const float TrackerSiamRPNPP::TRACK_LR = 0.5;

void TrackerSiamRPNPP::load_networks_instantly() {
	torch::Tensor z_crop = torch::zeros({ 1, 3, Tracker::EXEMPLAR_SIZE, Tracker::EXEMPLAR_SIZE }).cuda();
	torch::List<torch::Tensor> pre_zf = backbone_forward(z_crop);
	torch::List<torch::Tensor> zf = neck.forward({ pre_zf }).toTensorList();

	torch::Tensor x_crop = torch::zeros({ 1, 3, Tracker::INSTANCE_SIZE, Tracker::INSTANCE_SIZE }).cuda();
	torch::List<torch::Tensor> pre_xf = backbone_forward(x_crop);;
	torch::List<torch::Tensor> xf = neck.forward({ pre_xf }).toTensorList();

	for (int i = 0; i < rpns.size(); i++) {
		rpns[i].forward({ zf.get(i), xf.get(i) });
	}
}

torch::List<torch::Tensor> TrackerSiamRPNPP::backbone_forward(torch::Tensor crop) {
	torch::List<torch::Tensor> out;
	int nextUsedLayerIdx = 0;
	for (int i = 0; i < backbone.size() && nextUsedLayerIdx < BACKBONE_USED_LAYERS_NUM; i++) {
		crop = backbone[i].forward({ crop }).toTensor();
		if (i == BACKBONE_USED_LAYERS[nextUsedLayerIdx]) {
			out.push_back(crop);
			nextUsedLayerIdx++;
		}
	}
	return out;
}

cv::Rect TrackerSiamRPNPP::track(cv::Mat frame) {
	cv::Size frameSize = frame.size();

	// TODO: What are these?
	float s_z = calculate_s_z();
	float scale_z = Tracker::EXEMPLAR_SIZE / s_z;
	int s_x = round(s_z * Tracker::INSTANCE_SIZE / Tracker::EXEMPLAR_SIZE);
	torch::Tensor x_crop = get_subwindow(frame, Tracker::INSTANCE_SIZE, s_x);

	torch::List<torch::Tensor> pre_xf = backbone_forward(x_crop);
	torch::List<torch::Tensor> xf = neck.forward({ pre_xf }).toTensorList();

	torch::Tensor cls, loc;
	for (int i = 0; i < rpns.size(); i++) {
		std::vector<torch::IValue> res = rpns[i].forward({ zf.get(i), xf.get(i) }).toTuple()->elements();
		torch::Tensor c = res[0].toTensor().cuda();
		torch::Tensor l = res[1].toTensor().cuda();
		if (cls.numel() > 0) cls += c; else cls = c;
		if (loc.numel() > 0) loc += l; else loc = l;
	}
	cls /= (float)rpns.size();
	loc /= (float)rpns.size();

	torch::Tensor score = convert_score(cls);
	torch::Tensor pred_bbox = convert_bbox(loc);

	torch::Tensor s_c = change(
		sz(pred_bbox.narrow(0, 2, 1), pred_bbox.narrow(0, 3, 1)) / sz(torch::tensor(bounding_box.width * scale_z), torch::tensor(bounding_box.height * scale_z))
	);
	torch::Tensor r_c = change(
		torch::tensor((float)bounding_box.width / bounding_box.height) / (pred_bbox.narrow(0, 2, 1) / pred_bbox.narrow(0, 3, 1))
	);
	torch::Tensor penalty = torch::exp(-(r_c * s_c - 1) * Tracker::TRACK_PENALTY_K).t();
	torch::Tensor pscore = penalty * score * (1 - Tracker::TRACK_WINDOW_INFLUENCE) + window * Tracker::TRACK_WINDOW_INFLUENCE;
	int best_idx = torch::argmax(pscore).item().toInt();
	torch::Tensor bbox = pred_bbox.narrow(1, best_idx, 1) / scale_z;
	float lr = (penalty[best_idx] * score[best_idx]).item().toFloat() * Tracker::TRACK_LR;

	bounding_box.x = bbox[0].item().toFloat() + bounding_box.x + bounding_box.width / 2;
	bounding_box.y = bbox[1].item().toFloat() + bounding_box.y + bounding_box.height / 2;
	bounding_box.width = bounding_box.width * (1 - lr) + bbox[2].item().toFloat() * lr;
	bounding_box.height = bounding_box.height * (1 - lr) + bbox[3].item().toFloat() * lr;

	bounding_box.width = std::max(10, std::min(bounding_box.width, frameSize.width));
	bounding_box.height = std::max(10, std::min(bounding_box.height, frameSize.height));
	bounding_box.x = std::max(0, std::min(bounding_box.x, frameSize.width)) - bounding_box.width / 2;
	bounding_box.y = std::max(0, std::min(bounding_box.y, frameSize.height)) - bounding_box.height / 2;

	return bounding_box;
}