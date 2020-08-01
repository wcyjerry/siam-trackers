#include "TrackerSiamMask.h"

const int TrackerSiamMask::BACKBONE_USED_LAYERS[BACKBONE_USED_LAYERS_NUM] = { 2, 6, 7 };
const float TrackerSiamMask::MASK_THRESHOLD = 0.25f;

torch::List<torch::Tensor> TrackerSiamMask::backbone_forward(torch::Tensor crop) {
	torch::Tensor x = backbone_conv.forward({ crop }).toTensor();
	x = backbone_bn.forward({ x }).toTensor();
	torch::Tensor x_ = torch::relu(x);
	x = torch::max_pool2d(x_, 3, 2, 1);

	torch::List<torch::Tensor> out { x_ };
	int nextUsedLayerIdx = 0;
	for (int i = 0; i < backbone_layers.size(); i++) {
		x = backbone_layers[i].forward({ x }).toTensor();
		if (i == BACKBONE_USED_LAYERS[nextUsedLayerIdx]) {
			out.push_back(x);
			nextUsedLayerIdx++;
		}
	}
	return out;
}

void TrackerSiamMask::load_networks_instantly() {
	torch::Tensor z_crop = torch::zeros({ 1, 3, EXEMPLAR_SIZE, EXEMPLAR_SIZE }).cuda();
	torch::List<torch::Tensor> pre_zf = backbone_forward(z_crop);
	torch::List<torch::Tensor> zf = neck.forward({ persist_only_last(pre_zf) }).toTensorList();

	torch::Tensor x_crop = torch::zeros({ 1, 3, INSTANCE_SIZE, INSTANCE_SIZE }).cuda();
	torch::List<torch::Tensor> pre_xf = backbone_forward(x_crop);;
	torch::List<torch::Tensor> xf = neck.forward({ persist_only_last(pre_xf) }).toTensorList();

	rpn_head.forward({ zf.get(0), xf.get(0) }).toTuple()->elements();
}

cv::Mat TrackerSiamMask::track(cv::Mat frame) {
	cv::Size frame_size = frame.size();

	// TODO: What are these?
	float s_z = calculate_s_z();
	float scale_z = EXEMPLAR_SIZE / s_z;
	int s_x = round(s_z * INSTANCE_SIZE / EXEMPLAR_SIZE);
	torch::Tensor x_crop = get_subwindow(frame, INSTANCE_SIZE, s_x);

	torch::List<torch::Tensor> pre_xf = backbone_forward(x_crop);
	torch::List<torch::Tensor> pre_xf_head;
	for (int i = 0; i < pre_xf.size() - 1; i++) {
		pre_xf_head.push_back(pre_xf.get(i));
	}
	torch::List<torch::Tensor> xf = neck.forward({ persist_only_last(pre_xf) }).toTensorList();

	std::vector<torch::IValue> res_rpn = rpn_head.forward({ zf.get(0), xf.get(0) }).toTuple()->elements();
	torch::Tensor cls = res_rpn[0].toTensor().cuda();
	torch::Tensor loc = res_rpn[1].toTensor().cuda();

	torch::Tensor score = convert_score(cls);
	torch::Tensor pred_bbox = convert_bbox(loc);
	torch::Tensor penalty = get_penalty(scale_z, pred_bbox);
	int best_idx = get_best_idx(penalty, score);

	std::vector<int> pos = unravel_index(best_idx, { 5, SCORE_SIZE, SCORE_SIZE });
	int delta_x = pos[2], delta_y = pos[1];

	std::vector<torch::IValue> res_mask = mask_head.forward({ zf.get(0), xf.get(0) }).toTuple()->elements();
	torch::Tensor mask_corr_feature = res_mask[1].toTensor().cuda();

	torch::Tensor mask = refine_head.forward({
		pre_xf_head,
		mask_corr_feature,
		std::tuple<int, int>(delta_y, delta_x)
	}).toTensor().sigmoid().squeeze().view({ MASK_OUTPUT_SIZE, MASK_OUTPUT_SIZE }).to(torch::kCPU);

	float crop_box_x = (bounding_box.x + bounding_box.width / 2) - s_x / 2;
	float crop_box_y = (bounding_box.y + bounding_box.height / 2) - s_x / 2;

	float s = (float)s_x / INSTANCE_SIZE;
	float sub_box_x = crop_box_x + (delta_x - TRACK_BASE_SIZE / 2) * ANCHOR_STRIDE * s;
	float sub_box_y = crop_box_y + (delta_y - TRACK_BASE_SIZE / 2) * ANCHOR_STRIDE * s;
	float sub_box_size = s * EXEMPLAR_SIZE;

	s = MASK_OUTPUT_SIZE / sub_box_size;
	float back_box_x = -sub_box_x * s;
	float back_box_y = -sub_box_y * s;

	float a = (frame_size.width - 1) / (frame_size.width * s);
	float b = (frame_size.height - 1) / (frame_size.height * s);
	float c = -a * back_box_x;
	float d = -b * back_box_y;

	cv::Mat mapping = (cv::Mat_<float>(2, 3) << a, 0, c, 0, b, d);
	cv::Mat mat_mask = cv::Mat(MASK_OUTPUT_SIZE, MASK_OUTPUT_SIZE, CV_32F, mask.data_ptr());
	cv::Mat crop;
	cv::warpAffine(mat_mask, crop, mapping, frame_size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
	crop = (crop > MASK_THRESHOLD) * 255;

	cv::Mat empty_channel = crop * 0;
	std::vector<cv::Mat> channels { crop, empty_channel, empty_channel };
	cv::merge(channels, mat_mask);

	update_bbox(pred_bbox, best_idx, scale_z, penalty, score, frame_size);

	return mat_mask;
}
