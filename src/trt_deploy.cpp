#include <fstream>
#include <iostream>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/cudaarithm.hpp>
#include "trt_deploy.h"
#include "util.h"
#include <opencv2/core/cuda.hpp>
#include <thread>

namespace helmet
{

std::mutex m_mtx;
std::atomic_int m_thread_num = 0;

//SharedRef<Logger> m_logger = createSharedRef<Logger>();
//nvinfer1::IRuntime *m_runtime = nvinfer1::createInferRuntime(*m_logger);
//nvinfer1::ICudaEngine *m_engine = nullptr;


void Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept
{
	if (severity == Severity::kINFO||severity==Severity::kVERBOSE||severity==Severity::kWARNING){
		std::cout << msg << std::endl;
	}else{
		std::cerr<<msg<<std::endl;
	}
}

TrtDeploy::TrtDeploy(SharedRef<Config> &config, int gpuID)
{
	m_config = config;
	m_model_load_status = ModelLoadStatus::NON_LOADED;
	m_cuda_alloc_status = CudaMemAllocStatus::NON_ALLOC;
	m_curr_fps = 0.0f;
	m_gpu_id = gpuID;
	{
		std::lock_guard<std::mutex> lock(m_mtx);
		m_thread_num++;
	}
}

TrtDeploy::~TrtDeploy()
{
	cudaStreamSynchronize(m_stream);
	for (auto &i : m_device_ptr) {
		cudaFree(i);
	}
	for (auto &i : m_host_ptr) {
		cudaFreeHost(i);
	}
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
	{
		std::lock_guard<std::mutex> lock(m_mtx);
		m_thread_num--;
		std::cout << "Current remaining runtime ref count: " << m_thread_num << std::endl;
		if (m_thread_num == 0) {

			std::cout << "Free the runtime done..." << std::endl;
		}
	}
	std::cout << "Thread: " << std::this_thread::get_id() <<
			  " TensorRT Backend Deconstructed..." << std::endl;

}

void TrtDeploy::Infer(const cv::Mat &img, SharedRef<TrtResults> &result)
{
//        std::chrono::high_resolution_clock::time_point curr_time =
//                std::chrono::high_resolution_clock::now();
	if (!INIT_FLAG) {
		Init(m_config->MODEL_NAME);
		INIT_FLAG = true;
	}

	auto blob = createSharedRef<ImageBlob>();
	std::vector<cv::Mat> temp;
	temp.push_back(img);
	m_preprocessor->Run(temp, blob, m_thread_stream);
//        auto dur = std::chrono::high_resolution_clock::now() - curr_time;
//        curr_time = std::chrono::high_resolution_clock::now();
//        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
//        std::cout << "Thread: " << std::this_thread::get_id() << " Pre taken: " << ms << "ms" << std::endl;

	InferResults(blob, result);
}

void TrtDeploy::Init(const std::string &model_file)
{
	std::ifstream ai_model(model_file, std::ios::in | std::ios::binary);
	if (!ai_model) {
		std::cerr << "Read serialized file: " << model_file << " failed" << std::endl;
		m_model_load_status = ModelLoadStatus::LOADED_FAILED;
		return;
	}
	auto mSize = getFileSize(model_file);
	std::vector<char> buf(mSize);
	ai_model.read(&buf[0], mSize);
	ai_model.close();
	std::cout << "Model size: " << mSize << std::endl;
	m_config->MODEL_NAME = model_file;
	m_model_load_status = ModelLoadStatus::LOADED_SUCCESS;
	if (!m_preprocessor) {
		m_preprocessor = createSharedRef<Preprocessor>(m_config);
	}
	if (!m_postprocessor) {
		m_postprocessor = createSharedRef<Postprocessor>(m_config);
	}

	{
//		std::lock_guard<std::mutex> lock(m_mtx);
		initLibNvInferPlugins(m_logger.get(), "");
		if(!m_runtime)m_runtime = nvinfer1::createInferRuntime(*m_logger);
		if (!m_engine)
			m_engine = m_runtime->deserializeCudaEngine((void *)&buf[0],
														mSize);
		if (!m_execution_context)m_execution_context = m_engine->createExecutionContext();
		printf("Logger: %p; Runtime: %p; Engine: %p; Context: %p\n",
			   &m_logger, &m_runtime, &m_engine, &m_execution_context);
	}

	///note the target size should match the model input.
	std::vector<int> input_size;
	int in_size = 1;
	auto entry_num = m_config->INPUT_NAME.size();
	auto out_num = m_config->OUTPUT_NAMES.size();
	for (int i = 0; i < entry_num; ++i) {
		auto in_dims = m_engine->getTensorShape(m_config->INPUT_NAME[i].c_str());
		int curr = 1;
		for (int j = 0; j < in_dims.nbDims; ++j) {
			curr *= in_dims.d[j];
		}
		in_size *= curr;
		input_size.push_back(curr);
	}
	m_host_size.resize(out_num, 0);
	m_host_ptr.resize(out_num, nullptr);
	m_device_ptr.resize(entry_num + out_num, nullptr);//with input pointer, thus+1.
	for (int i = 0; i < out_num; ++i) {
		auto out_dims_i = m_engine->getTensorShape(m_config->OUTPUT_NAMES[i].c_str());
		int out_size = 1;
		for (int j = 0; j < out_dims_i.nbDims; ++j) {
			if (out_dims_i.d[j] > 0)out_size *= out_dims_i.d[j];
			else out_size *= -out_dims_i.d[j];
		}
		cudaMallocHost(&m_host_ptr[i], out_size * sizeof(float));
		m_host_size[i] = out_size;
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
	for (auto i = entry_num; i < entry_num + out_num; ++i) {
		state = cudaMalloc(&m_device_ptr[i],
						   m_host_size[i - entry_num] * sizeof(float));
		if (state) {
			std::cout << "Allocate memory failed" << std::endl;
			m_cuda_alloc_status = CudaMemAllocStatus::ALLOC_FAILED;
			return;
		}
	}
	m_cuda_alloc_status = CudaMemAllocStatus::ALLOC_SUCCESS;

	if (!m_thread_stream) {
		m_thread_stream = createSharedRef<cv::cuda::Stream>(cudaStreamNonBlocking);
	}
	if (!m_stream) {
		m_stream = static_cast<cudaStream_t>(m_thread_stream->cudaPtr());
	}

	for (int i = 0; i < entry_num; ++i) {
		m_execution_context->setTensorAddress(m_config->INPUT_NAME[i].c_str(),
											  m_device_ptr[i]);
	}
	for (int i = 0; i < out_num; ++i) {
		m_execution_context->setTensorAddress(m_config->OUTPUT_NAMES[i].c_str(),
											  m_device_ptr[i + entry_num]);

	}

	int w = m_config->TARGET_SIZE[m_config->TARGET_SIZE.size() - 1];
	int h = m_config->TARGET_SIZE[m_config->TARGET_SIZE.size() - 2];
	for (int i = 0; i < m_config->INPUT_NAME.size(); ++i) {
		auto *ptr = (float *)m_device_ptr[i];
		if (i == 0) {
			for (int k = 0; k < m_config->TRIGGER_LEN; ++k) {
				cv::cuda::GpuMat temp(1, 2, CV_32FC1, ptr + k * 2);
				m_im_shape.emplace_back(temp);
			}
		}
		else if (i == 1) {
			for (int k = 0; k < m_config->TRIGGER_LEN; ++k) {
				for (int j = 0; j < 3; ++j) {
					m_cv_data.emplace_back(cv::Size(w, h),
										   CV_32FC1, ptr + j * w * h + k * 3 * w * h);
				}
			}
		}
		else if (i == 2) {
			for (int k = 0; k < m_config->TRIGGER_LEN; ++k) {
				cv::cuda::GpuMat temp(cv::Size(2, 1), CV_32FC1, ptr + k * 2);
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
	cv::Mat img = cv::Mat::ones(cv::Size(m_config->INPUT_SHAPE[m_config->INPUT_SHAPE.size() - 1],
										 m_config->INPUT_SHAPE[m_config->INPUT_SHAPE.size() - 2]), CV_8UC3);
	for (int i = 0; i < 10; i++)
		Infer(img, res);
}

void TrtDeploy::InferResults(SharedRef<ImageBlob> &data, SharedRef<TrtResults> &res)
{

	res->Clear();

	for (int i = 0; i < data->m_gpu_data.size(); ++i) {
		cv::cuda::split(data->m_gpu_data[i], &m_cv_data[3 * i], *m_thread_stream);
	}
	std::vector<float> shape = {static_cast<float>(m_config->TARGET_SIZE[0]),
								static_cast<float>(m_config->TARGET_SIZE[1])};
	std::vector<float> scale = {1.0f, 1.0f};
	for (auto &item : m_im_shape) {
		item.upload(shape, *m_thread_stream);
	}
	for (auto &item : m_scale_factor) {
		item.upload(scale, *m_thread_stream);
	}
//        std::chrono::high_resolution_clock::time_point curr_time =
//                std::chrono::high_resolution_clock::now();
	m_execution_context->enqueueV3(m_stream);
//        auto dur = std::chrono::high_resolution_clock::now() - curr_time;
//        curr_time = std::chrono::high_resolution_clock::now();
//        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
//        std::cout << "Thread: " << std::this_thread::get_id() << " enqueue taken: " << ms << "ms" << std::endl;

	auto entry = m_config->INPUT_NAME.size();
	for (int i = 0; i < m_config->OUTPUT_NAMES.size(); ++i) {
		auto state = cudaMemcpyAsync(m_host_ptr[i], m_device_ptr[i + entry],
									 m_host_size[i] * sizeof(float),
									 cudaMemcpyDeviceToHost, m_stream);
		if (state) {
			std::cout << "Transmit to host failed." << std::endl;
		}
	}
//        cudaStreamSynchronize(m_stream);
	for (int i = 0; i < m_host_size.size(); ++i) {
		std::vector<float> temp(m_host_size[i], 0.0);
		memcpy(temp.data(), m_host_ptr[i], sizeof(float) * m_host_size[i]);
		res->Set(std::make_pair(m_config->OUTPUT_NAMES[i], temp));
	}
}

void TrtDeploy::Postprocessing(const SharedRef<TrtResults> &res, cv::Mat &img, int &alarm)
{
	m_postprocessor->Run(res, img, alarm);
}

}
