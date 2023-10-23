#include "preprocess_util.hpp"
#include "preprocessor.h"

namespace helmet
{

#ifdef PREPROCESS_GPU

void PreprocessorFactory::Init()
{
	assert(m_config->N_STD[0] > 0.0f && m_config->N_STD[1] > 0.0f && m_config->N_STD[2] > 0.0f);
	INIT_FLAG = true;
}

PreprocessorFactory::PreprocessorFactory(SharedRef<Config>& config)
{
	m_config = config;

	m_workers["TopDownEvalAffine"] = new TopDownEvalAffine(config);
	m_workers["Resize"] = new Resize(config);
	m_workers["LetterBoxResize"] = new LetterBoxResize(config);
	m_workers["NormalizeImage"] = new NormalizeImage(config);
	m_workers["PadStride"] = new PadStride(config);
	m_workers["Permute"] = new Permute(config);

	if (!m_stream) {
		m_stream = createSharedRef<cv::cuda::Stream>();
	}
}

void PreprocessorFactory::CvtForGpuMat(const std::vector<cv::Mat> &input,
									   std::vector<cv::cuda::GpuMat> &frames, int &num)
{

	num = (int)input.size();
	for (int i = 0; i < num; ++i) {
		frames[i].upload(input[i], *m_stream);
	}
	m_stream->waitForCompletion();
	for (int i = 0; i < num; ++i) {
		frames[i].convertTo(frames[i], cv::COLOR_BGR2RGB, *m_stream);
        frames[i].convertTo(frames[i],CV_32FC3,1.0,0.0,*m_stream);
	}
	m_stream->waitForCompletion();
}

void PreprocessorFactory::Run(const std::vector<cv::Mat> &input, SharedRef<ImageBlob> &output)
{
	if (!INIT_FLAG) {
		Init();
		INIT_FLAG = true;
		if (m_config->INPUT_SHAPE[m_config->INPUT_SHAPE.size() - 1] != input[0].cols) {
			m_config->INPUT_SHAPE[m_config->INPUT_SHAPE.size() - 1] = input[0].cols;
			SCALE_W = (float)m_config->TARGET_SIZE[1] / (float)input[0].cols;
			std::cout << "Input shape width in config file is not same as data width..." << std::endl;
		}
		if (m_config->INPUT_SHAPE[m_config->INPUT_SHAPE.size() - 2] != input[0].rows) {
			m_config->INPUT_SHAPE[m_config->INPUT_SHAPE.size() - 2] = input[0].rows;
			SCALE_H = (float)m_config->TARGET_SIZE[0] / (float)input[0].rows;
			std::cout << "Input shape height in config file is not same as data height..." << std::endl;
		}
	}
	std::vector<cv::cuda::GpuMat> gpu(input.size());
	int num = 0;
	CvtForGpuMat(input, gpu, num);
	for (const auto &i : m_config->PIPELINE_TYPE) {
		m_workers[i]->Run(gpu);
	}

    if (output->im_shape_.empty()) {
        output->im_shape_ = m_config->INPUT_SHAPE;
    }
    if (output->in_net_shape_.empty()) {
        output->in_net_shape_ = output->im_shape_;
        output->in_net_shape_[output->im_shape_.size() - 1] = m_config->TARGET_SIZE[1];
        output->in_net_shape_[output->im_shape_.size() - 2] = m_config->TARGET_SIZE[0];
    }
    if (output->scale_factor_.empty()) {
        output->scale_factor_.resize(2);
        output->scale_factor_[0] = SCALE_H;
        output->scale_factor_[1] = SCALE_W;
    }
    output->m_gpu_data = gpu;
}

PreprocessorFactory::~PreprocessorFactory()
{
	for(auto& [name,item]:m_workers){
		delete item;
		item = nullptr;
	}
}

void Preprocessor::Run(const std::vector<cv::Mat> &input, SharedRef<ImageBlob> &output)
{
	if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
		std::cerr << "Your OpenCV does not support CUDA!" << std::endl;
		std::cerr << "Please install CUDA version OpenCV! "
					 "See: https://towardsdev.com/installing-opencv-4-with-cuda-in-ubuntu-20-04-fde6d6a0a367"
				  << std::endl;
	}
	if (!m_preprocess_factory) {
		m_preprocess_factory = createSharedRef<PreprocessorFactory>(m_config);
	}
	m_preprocess_factory->Run(input, output);
}


#endif

}
