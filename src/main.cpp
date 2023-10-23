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

int main(int argc, char **argv) {

    //prepare the input data.
    auto in_path = std::filesystem::path("/home/wgf/Downloads/datasets/Anquanmao/helmet-live/09-38.mp4");
    cv::VideoCapture cap(in_path);
    cv::VideoWriter vw;
    std::filesystem::path output_path = in_path.parent_path() / (in_path.stem().string() + ".result.mp4");
    vw.open(output_path,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            cap.get(cv::CAP_PROP_FPS),
            cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH),
					 cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

    cv::Mat img;
	bool init = false;
	cvModel *ptr  = nullptr;
    while (cap.isOpened()) {
		cap.read(img);
		if(!init){
			ptr = Allocate_Algorithm(img,IA_TYPE_PEOPLEHELME_DETECTION,0);
			SetPara_Algorithm(ptr,IA_TYPE_PEOPLEHELME_DETECTION);
			UpdateParams_Algorithm(ptr);
			init = true;
		}
		if(img.cols==0||img.rows==0)break;
		Process_Algorithm(ptr,img);
		vw.write(img.clone());
    }
	cap.release();
	vw.release();
	Destroy_Algorithm(ptr);

    return 0;
}