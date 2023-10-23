#include "model.h"
#include "config.h"
#include "trt_deploy.h"
#include "trt_deployresult.h"

namespace helmet{

class InferModel {
public:
	explicit InferModel(int gpuID, SharedRef<Config>& config) {
		m_config = config;
		mDeploy = createSharedRef<TrtDeploy>(config,gpuID);
		mResult = createSharedRef<TrtResults>(config);
	}

public:
	SharedRef<TrtDeploy> mDeploy;
	SharedRef<TrtResults> mResult;
	SharedRef<Config> m_config;
};

static void *GenModel(int gpuID,SharedRef<Config> config) {
	auto *model = new InferModel(gpuID,config);
	return reinterpret_cast<void *>(model);
}

cvModel *Allocate_Algorithm(cv::Mat &input_frame, int algID, int gpuID) {
	std::string file;
	if(Util::checkFileExist("./helmet_detection.yaml"))
		file = "./helmet_detection.yaml";
	else if(Util::checkFileExist("../config/helmet_detection.yaml")){
		file = "../config/helmet_detection.yaml";
	}else{
		std::cout<<"Cannot find YAML file!"<<std::endl;
	}
	auto config = createSharedRef<Config>(0, nullptr, file);
	config->INPUT_SHAPE[config->INPUT_SHAPE.size()-1] = input_frame.cols;
	config->INPUT_SHAPE[config->INPUT_SHAPE.size()-2] = input_frame.rows;
	auto *ptr = new cvModel();
	ptr->FrameNum = 0;
	ptr->Frameinterval = 0;
	ptr->countNum = 0;
	ptr->width = input_frame.cols;
	ptr->height = input_frame.rows;
	ptr->iModel = GenModel(gpuID,config);
	return ptr;
}

void SetPara_Algorithm(cvModel *pModel, int algID) {
	//todo: implement this
}

void UpdateParams_Algorithm(cvModel *pModel) {
	//todo: implement this
}

void Process_Algorithm(cvModel *pModel, cv::Mat &input_frame) {
	auto model = reinterpret_cast<InferModel *>(pModel->iModel);
	auto roi = pModel->p;

	auto config = model->m_config;
	model->mDeploy->Infer(input_frame, model->mResult);
	model->mDeploy->Postprocessing(model->mResult, input_frame,pModel->alarm);
	cv::line(input_frame, cv::Point(roi[0].x, roi[0].y),
			 cv::Point(roi[1].x, roi[1].y), cv::Scalar(255, 0, 0),
			 config->BOX_LINE_WIDTH);
	cv::line(input_frame, cv::Point(roi[1].x, roi[1].y),
			 cv::Point(roi[2].x, roi[2].y), cv::Scalar(255, 0, 0),
			 config->BOX_LINE_WIDTH);
	cv::line(input_frame, cv::Point(roi[2].x, roi[2].y),
			 cv::Point(roi[3].x, roi[3].y), cv::Scalar(255, 0, 0),
			 config->BOX_LINE_WIDTH);
	cv::line(input_frame, cv::Point(roi[0].x, roi[0].y),
			 cv::Point(roi[3].x, roi[3].y), cv::Scalar(255, 0, 0),
			 config->BOX_LINE_WIDTH);
}

void Destroy_Algorithm(cvModel *pModel) {
	if (pModel->iModel){
		auto model = reinterpret_cast<InferModel *>(pModel->iModel);
		delete model;
		model = nullptr;
	}
	if (pModel) {
		delete pModel;
		pModel = nullptr;
	}
}
}

