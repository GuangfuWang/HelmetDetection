#include "util.h"
#include <sys/stat.h>
#include <iostream>
#include <valarray>

namespace helmet
{

thread_local std::chrono::high_resolution_clock::time_point Util::mTic;

bool checkDirExist(const std::string &dir)
{
	///note this feature require c++ 17 and above.
	return std::filesystem::exists(dir);
}

bool checkFileExist(const std::string &file)
{
	///note this feature require c++ 17 and above.
	return std::filesystem::exists(file);
}

void Util::tic()
{
	mTic = std::chrono::high_resolution_clock::now();
}

long Util::toc()
{
	auto dur = std::chrono::high_resolution_clock::now() - mTic;
	mTic = std::chrono::high_resolution_clock::now();
	long ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	return ms;
}

long getFileSize(const std::string &file)
{
	if (!checkFileExist(file)) {
		std::cerr << "File: " << file << " not exists..." << std::endl;
		return -1;
	}
	struct stat statBuf;
	stat(file.c_str(), &statBuf);
	return statBuf.st_size;
}

int Util::cvtStr2Int(const char *arr)
{
	return std::atoi(arr);
}
void Util::softmax(std::vector<float> &in)
{
	float total = 0.0f;
	for (auto &e : in) {
		auto curr = std::exp(e);
		e = curr;
		total += curr;
	}
	for (auto &e : in) {
		e /= total;
	}
}
std::vector<std::string> parseNames(const std::string &names, char delim)
{
	std::stringstream ss(names);
	std::string item;
	std::vector<std::string> elems;
	while (std::getline(ss, item, delim)) {
		if (!item.empty()) {
			elems.push_back(item);
		}
	}
	return elems;
}

void plotBox(cv::Mat &img, int x0, int y0, int x1, int y1, std::vector<unsigned char> color, int thickness)
{
	cv::line(img, cv::Point(x0, y0),
			 cv::Point(x1, y0), cv::Scalar(color[0], color[1], color[2]),
			 thickness);
	cv::line(img, cv::Point(x0, y0),
			 cv::Point(x0, y1), cv::Scalar(color[0], color[1], color[2]),
			 thickness);
	cv::line(img, cv::Point(x0, y1),
			 cv::Point(x1, y1), cv::Scalar(color[0], color[1], color[2]),
			 thickness);
	cv::line(img, cv::Point(x1, y0),
			 cv::Point(x1, y1), cv::Scalar(color[0], color[1], color[2]),
			 thickness);
}
int round2int(float num)
{
	if (num > 0)
		return num - int(num) >= 0.5 ? int(num) + 1 : int(num);
	else
		return -num - int(-num) >= 0.5 ? -(int(-num) + 1) : -int(-num);

}
}
