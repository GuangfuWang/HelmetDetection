#include <thread>
#include <future>
#include "config.h"
#include "trt_deploy.h"
#include "trt_deployresult.h"
#include "model.h"

using namespace helmet;

/**
 * @example
 * @param argc number of input params, at least 1.
 * @param argv params lists
 * @return
 */

int TEST_THREADS = 1;

void process_video(int thread_id, const std::string &file)
{
	//prepare the input data.
	if (!checkFileExist(file)) {
		std::cerr << "The video file is not exist..." << std::endl;
		return;
	}
//    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator (cv::cuda::HostMem::AllocType::PAGE_LOCKED));

	auto in_path = std::filesystem::path(file);
	cv::VideoCapture cap(in_path);
	cv::VideoWriter vw;

	std::filesystem::path
		output_path = in_path.parent_path() / (in_path.stem().string() + std::to_string(thread_id) + ".mp4");
	vw.open(output_path,
			cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
			cap.get(cv::CAP_PROP_FPS),
			cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH),
					 cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

	cv::Mat img;
	bool init = false;
	cvModel *models = nullptr;
	std::chrono::high_resolution_clock::time_point curr_time;
	while (cap.isOpened()) {
		cap.read(img);
		if (!init) {
			models = Allocate_Algorithm(img, IA_TYPE_PEOPLEHELME_DETECTION, 0);
			SetPara_Algorithm(models, IA_TYPE_PEOPLEHELME_DETECTION);
			UpdateParams_Algorithm(models);
			init = true;
		}
		if (img.cols == 0 || img.rows == 0)break;
		curr_time =
			std::chrono::high_resolution_clock::now();
		Process_Algorithm(models, img);
		auto dur = std::chrono::high_resolution_clock::now() - curr_time;
		curr_time = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << "Thread: " << std::this_thread::get_id() << " Cpu: " << sched_getcpu() << " taken: " << ms << "ms"
				  << std::endl;
		vw.write(img.clone());
	}
	cap.release();
	vw.release();
	Destroy_Algorithm(models);

}

int main(int argc, char **argv)
{
	bool enable_cpu_affinity = false;
	if (argc > 1) {
		TEST_THREADS = std::atoi(argv[1]);
	}
	if (argc > 2) {
		int temp = std::atoi(argv[2]);
		if (temp)enable_cpu_affinity = true;
	}
	std::vector<std::thread> threads(TEST_THREADS);
	std::vector<std::string> files(TEST_THREADS);
	std::string base = "/home/wgf/Downloads/datasets/Anquanmao/helmet-live/";
	for (int i = 0; i < TEST_THREADS; ++i) {
		files[i] = base + std::to_string(i) + ".mp4";
//        files[i] = "/home/wgf/Downloads/datasets/Anquanmao/helmet-live/multithread.mp4";
		threads[i] = std::thread(process_video, i, files[i]);
		if (enable_cpu_affinity) {
			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			CPU_SET(i, &cpuset);
			int rc = pthread_setaffinity_np(threads[i].native_handle(),
											sizeof(cpu_set_t), &cpuset);
			if (rc != 0) {
				std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
			}
		}
	}
	for (int i = 0; i < TEST_THREADS; ++i) {
		threads[i].join();
	}

	return 0;
}