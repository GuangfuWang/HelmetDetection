#include <fstream>
#include <iostream>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/cudaarithm.hpp>
#include "trt_deploy.h"
#include "util.h"
#include <opencv4/opencv2/core/cuda.hpp>
namespace helmet
{

void Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept
{
	if (severity != Severity::kINFO) {
		std::cerr << msg << std::endl;
	}
}

thread_local bool TrtDeploy::INIT_FLAG = false;

TrtDeploy::TrtDeploy(int gpuID)
{
	m_model_load_status = ModelLoadStatus::NON_LOADED;
	m_cuda_alloc_status = CudaMemAllocStatus::NON_ALLOC;
	m_curr_fps = 0.0f;
	cv::cuda::setDevice(gpuID);
	cudaSetDevice(gpuID);
}

TrtDeploy::~TrtDeploy()
{
	cudaStreamSynchronize(m_stream);
	for (int i = 0; i <= Config::OUTPUT_NAMES.size(); ++i) {
		cudaFree(m_device_ptr[i]);
	}
	cudaStreamDestroy(m_stream);

	if (m_execution_context) {
		delete m_execution_context;
		m_execution_context = nullptr;
	}
	if (m_engine) {
		delete m_engine;
		m_engine = nullptr;
	}
	if (m_runtime) {
		delete m_runtime;
		m_runtime = nullptr;
	}
	std::cout << "TensorRT Deploy Backend Deconstructed..." << std::endl;
}

void TrtDeploy::Infer(const cv::Mat &img, SharedRef<TrtResults> &result)
{
	if (Config::TIMING) {
		Util::tic();
	}
	if (!INIT_FLAG) {
		Init(Config::MODEL_NAME);
		INIT_FLAG = true;
	}
	auto blob = createSharedRef<ImageBlob>();
	std::vector<cv::Mat> temp;
	temp.push_back(img);
	m_preprocessor->Run(temp, blob);
	InferResults(blob, result);
	if (Config::TIMING) {
		auto timing = Util::toc();
		if (timing > 0) {
			m_curr_fps = 1000.0f / (float)timing;
			std::cout << "Current FPS: " << m_curr_fps << std::endl;
		}
	}

}

void TrtDeploy::Init(const std::string &model_file)
{
	std::ifstream ai_model(model_file, std::ios::in | std::ios::binary);
	if (!ai_model) {
		std::cerr << "Read serialized file: " << model_file << " failed" << std::endl;
		m_model_load_status = ModelLoadStatus::LOADED_FAILED;
		return;
	}
	auto mSize = Util::getFileSize(model_file);
	std::vector<char> buf(mSize);
	ai_model.read(&buf[0], mSize);
	ai_model.close();
	std::cout << "Model size: " << mSize << std::endl;
	Config::MODEL_NAME = model_file;
	m_model_load_status = ModelLoadStatus::LOADED_SUCCESS;
	if (!m_logger) {
		m_logger = createSharedRef<Logger>();
	}
	if (!m_preprocessor) {
		m_preprocessor = createSharedRef<Preprocessor>();
	}
	if (!m_postprocessor) {
		m_postprocessor = createSharedRef<Postprocessor>();
	}
	if (!m_runtime) {
		m_runtime = nvinfer1::createInferRuntime(*m_logger);
		initLibNvInferPlugins(m_logger.get(), "");
		m_engine = m_runtime->deserializeCudaEngine((void *)&buf[0], mSize);
		m_execution_context = m_engine->createExecutionContext();

	}

	///note the target size should match the model input.
	std::vector<int> input_size;
	int in_size = 1;
	auto entry_num = Config::INPUT_NAME.size();
	auto out_num = Config::OUTPUT_NAMES.size();
	for (int i = 0; i < entry_num; ++i) {
		auto in_dims = m_engine->getTensorShape(Config::INPUT_NAME[i].c_str());
		int curr = 1;
		for (int i = 0; i < in_dims.nbDims; ++i) {
			curr *= in_dims.d[i];
		}
		in_size *= curr;
		input_size.push_back(curr);
	}
	m_output_state.resize(out_num);
	m_device_ptr.resize(entry_num + out_num);//with input pointer, thus+1.
	for (int i = 0; i < out_num; ++i) {
		auto out_dims_i = m_engine->getTensorShape(Config::OUTPUT_NAMES[i].c_str());
		int out_size = 1;
		for (int j = 0; j < out_dims_i.nbDims; ++j) {
			if (out_dims_i.d[j] > 0)out_size *= out_dims_i.d[j];
			else out_size *= -out_dims_i.d[j];
		}
		m_output_state[i].resize(out_size, 0);
//		printf("Output Size: %d\tNumber dims: %d\n", out_size,out_dims_i.nbDims);
	}

	cudaError_t state;
	for (int i = 0; i < entry_num; ++i) {
		state = cudaMalloc(&m_device_ptr[i],
						   input_size[i] * sizeof(float));
		if (state) {
			std::cout << "Allocate memory failed" << std::endl;
			m_cuda_alloc_status = CudaMemAllocStatus::ALLOC_FAILED;
			return;
		}
	}
	for (int i = entry_num; i < entry_num + out_num; ++i) {
		state = cudaMalloc(&m_device_ptr[i],
						   m_output_state[i - entry_num].size() * sizeof(float));
		if (state) {
			std::cout << "Allocate memory failed" << std::endl;
			m_cuda_alloc_status = CudaMemAllocStatus::ALLOC_FAILED;
			return;
		}
	}
	m_cuda_alloc_status = CudaMemAllocStatus::ALLOC_SUCCESS;
	state = cudaStreamCreate(&m_stream);
	if (state) {
		std::cout << "Create stream failed" << std::endl;
		return;
	}
	for (int i = 0; i < entry_num; ++i) {
		m_execution_context->setTensorAddress(Config::INPUT_NAME[i].c_str(),
											  m_device_ptr[i]);
	}
	for (int i = 0; i < out_num; ++i) {
		m_execution_context->setTensorAddress(Config::OUTPUT_NAMES[i].c_str(),
											  m_device_ptr[i + entry_num]);

	}

	int w = Config::TARGET_SIZE[Config::TARGET_SIZE.size() - 1];
	int h = Config::TARGET_SIZE[Config::TARGET_SIZE.size() - 2];
	for (int i = 0; i < Config::INPUT_NAME.size(); ++i) {
		auto *ptr = (float *)m_device_ptr[i];
		if (i == 0) {
			for (int k = 0; k < Config::TRIGGER_LEN; ++k) {
				cv::cuda::GpuMat temp(1, 2, CV_32FC1, ptr + k * 2);
				m_im_shape.emplace_back(temp);
			}
		}
		else if (i == 1) {
			for (int k = 0; k < Config::TRIGGER_LEN; ++k) {
				for (int j = 0; j < 3; ++j) {
					m_cv_data.emplace_back(cv::Size(w, h),
										   CV_32FC1, ptr + j * w * h + k * 3 * w * h);
				}
			}
		}
		else if (i == 2) {
			for (int k = 0; k < Config::TRIGGER_LEN; ++k) {
				cv::cuda::GpuMat temp(cv::Size(2, 1), CV_32FC1, ptr+k*2);
				m_scale_factor.emplace_back(temp);
			}
		}
		else {
			std::cerr << "Not supported inputs..." << std::endl;
		}
	}
}

TrtDeploy::ModelLoadStatus TrtDeploy::LoadStatus()
{
	return m_model_load_status;
}

TrtDeploy::CudaMemAllocStatus TrtDeploy::MemAllocStatus()
{
	return m_cuda_alloc_status;
}

void TrtDeploy::Warmup(SharedRef<TrtResults> &res)
{
	cv::Mat img = cv::Mat::ones(cv::Size(Config::INPUT_SHAPE[Config::INPUT_SHAPE.size() - 1],
										 Config::INPUT_SHAPE[Config::INPUT_SHAPE.size() - 2]), CV_8UC3);
	for (int i = 0; i < 10; i++)
		Infer(img, res);
}

void TrtDeploy::InferResults(SharedRef<ImageBlob> &data, SharedRef<TrtResults> &res)
{
	res->Clear();

	for (int i = 0; i < data->m_gpu_data.size(); ++i) {
		cv::cuda::split(data->m_gpu_data[i], &m_cv_data[3 * i]);
	}
	std::vector<float> t = {static_cast<float>(Config::TARGET_SIZE[0]),
							static_cast<float>(Config::TARGET_SIZE[1])};
	for (auto &item : m_im_shape) {
		item.upload(t);
	}
	std::vector<float> ss = {1.0f, 1.0f};
	for (auto &item : m_scale_factor) {
		item.upload(ss);
	}
	m_execution_context->enqueueV3(m_stream);
	auto entry = Config::INPUT_NAME.size();

	for (int i = 0; i < Config::OUTPUT_NAMES.size(); ++i) {
		auto state = cudaMemcpyAsync(&m_output_state[i][0], m_device_ptr[i + entry],
									 m_output_state[i].size() * sizeof(float),
									 cudaMemcpyDeviceToHost, m_stream);
		if (state) {
			std::cout << "Transmit to host failed." << std::endl;
		}
	}
//	cudaStreamSynchronize(m_stream);
	//warp the state to our res.
	int idx = 0;
	for (auto &name : Config::OUTPUT_NAMES) {
		res->Set(std::make_pair(name, m_output_state[idx]));
		idx++;
	}
}

void TrtDeploy::Postprocessing(const SharedRef<TrtResults> &res, cv::Mat &img,int& alarm)
{
	m_postprocessor->Run(res, img,alarm);
}

}