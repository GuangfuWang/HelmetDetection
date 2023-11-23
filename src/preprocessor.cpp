#include <thread>
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

PreprocessorFactory::PreprocessorFactory(SharedRef<Config> &config, SharedRef<cv::cuda::Stream> &stream)
{
	m_config = config;

	m_workers["TopDownEvalAffine"] = new TopDownEvalAffine(config, stream);
	m_workers["Resize"] = new Resize(config, stream);
	m_workers["LetterBoxResize"] = new LetterBoxResize(config, stream);
	m_workers["NormalizeImage"] = new NormalizeImage(config, stream);
	m_workers["PadStride"] = new PadStride(config, stream);
	m_workers["Permute"] = new Permute(config, stream);

	if (!m_stream) {
		m_stream = stream;
		m_cuda_stream = static_cast<cudaStream_t>(m_stream->cudaPtr());
	}
}

void PreprocessorFactory::CvtForGpuMat(const std::vector<cv::Mat> &input,
									   std::vector<cv::cuda::GpuMat> &frames, int &num)
{

	num = (int)input.size();
	for (int i = 0; i < num; ++i) {
		frames[i].upload(input[i], *m_stream);
		frames[i].convertTo(frames[i], cv::COLOR_BGR2RGB, *m_stream);
		frames[i].convertTo(frames[i], CV_32FC3, 1.0, 0.0, *m_stream);
	}
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
		m_gpu_data.resize(input.size());
		auto ss = input[0].total() * input[0].elemSize();
		for (auto &i : m_gpu_data) {
			void *ptr = nullptr;
			cudaMalloc(&ptr, ss);
			i = cv::cuda::GpuMat(input[0].rows, input[0].cols, input[0].type(), ptr);
		}
		m_input_paged_mat.resize(input.size());
		m_input.resize(input.size());
		for (int i = 0; i < input.size(); i++) {
			cudaMallocHost(&m_input_paged_mat[i], ss);
			m_input[i] = cv::Mat(cv::Size(input[0].cols, input[0].rows),
								 input[0].type(), m_input_paged_mat[i]);

		}

	}
	for (int i = 0; i < input.size(); i++) {
		memcpy(m_input_paged_mat[i], input[i].data, input[i].total() * input[i].elemSize());
	}
	int num = 0;
	CvtForGpuMat(m_input, m_gpu_data, num);

	for (const auto &i : m_config->PIPELINE_TYPE) {
		m_workers[i]->Run(m_gpu_data);
	}

	output->m_gpu_data = this->m_gpu_data;
}

PreprocessorFactory::~PreprocessorFactory()
{
	for (auto &[name, item] : m_workers) {
		delete item;
		item = nullptr;
	}
	for(auto &i:m_input_paged_mat){
		cudaFreeHost(i);
	}
}

void Preprocessor::Run(const std::vector<cv::Mat> &input,
					   SharedRef<ImageBlob> &output,
					   SharedRef<cv::cuda::Stream> &stream)
{
	if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
		std::cerr << "Your OpenCV does not support CUDA!" << std::endl;
		std::cerr << "Please install CUDA version OpenCV! "
					 "See: https://towardsdev.com/installing-opencv-4-with-cuda-in-ubuntu-20-04-fde6d6a0a367"
				  << std::endl;
	}
	if (!m_preprocess_factory) {
		m_preprocess_factory = createSharedRef<PreprocessorFactory>(m_config, stream);
	}

	m_preprocess_factory->Run(input, output);

}


#endif

}
