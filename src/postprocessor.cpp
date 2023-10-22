#include <opencv2/imgproc.hpp>
#include "postprocessor.h"
#include "config.h"
#include <cmath>
namespace helmet
{

thread_local bool Postprocessor::INIT_FLAG = false;

void HelmetDetectionPost::Run(const SharedRef<TrtResults> &res, cv::Mat &img,int &alarm)
{
	//our simple program will only draw letters on top of images.
	auto flag = static_cast<PostProcessFlag>(Config::POST_MODE);
	assert(flag == PostProcessFlag::DRAW_BOX_LETTER);
	alarm = 0;

	std::vector<float> dets;
	std::vector<float> num_dets;
	res->Get(Config::OUTPUT_NAMES[0], dets);
	res->Get(Config::OUTPUT_NAMES[1], num_dets);

//	std::vector<float> num_dets;
//	std::vector<float> dets_box;
//	std::vector<float> dets_scores;
//	std::vector<float> dets_classes;
//	res->Get(Config::OUTPUT_NAMES[0], num_dets);
//	res->Get(Config::OUTPUT_NAMES[1], dets_box);
//	res->Get(Config::OUTPUT_NAMES[2], dets_scores);
//	res->Get(Config::OUTPUT_NAMES[3], dets_classes);
	float scale_x = (float)img.cols / (float)Config::TARGET_SIZE[1];
	float scale_y = (float)img.rows / (float)Config::TARGET_SIZE[0];

	std::vector<Box> curr;
//	printf("Num of detections: %f\n",num_dets[0]);
	for (int j = 0; j < 100; ++j) {
		Box b;
//		b.class_id = Util::round2int(dets_classes[j]);
//		b.score = dets_scores[j];
//		b.x_min = (int)(dets_box[0+j*4] * scale_x);
//		b.y_min = (int)(dets_box[1+j*4] * scale_y);
//		b.x_max = (int)(dets_box[2+j*4] * scale_x);
//		b.y_max = (int)(dets_box[3+j*4] * scale_y);
		b.class_id = Util::round2int(dets[0+j*6]);
		b.score = dets[1+j*6];
		b.x_min = (int)(dets[2+j*6] * scale_x);
		b.y_min = (int)(dets[3+j*6] * scale_y);
		b.x_max = (int)(dets[4+j*6] * scale_x);
		b.y_max = (int)(dets[5+j*6] * scale_y);

//		if (b.score > 0) {
//			std::cout <<"class_id: " <<b.class_id
//					  << "; score: " << b.score
//					  << "; x0: " << b.x_min << "; y0: " << b.y_min
//					  << "; x1: " << b.x_max << "; y1: " << b.y_max
//					  << std::endl;
//		}
		curr.push_back(b);
	}
	auto b = curr;
	///@note the putText method does not have GPU version since it quite slow running on GPU for per pixel ops.
	for (int k = 0; k < b.size(); ++k) {
		if(b[k].class_id>1)continue;
		if (b[k].score > Config::SCORE_THRESHOLD) {
			std::vector<unsigned char> box_color;
			box_color.resize(3);
			std::vector<unsigned char> text_color;
			box_color.resize(3);
			if(b[k].class_id==Config::TARGET_CLASS){
				box_color = Config::ALARM_BOX_COLOR;
				text_color = Config::ALARM_TEXT_COLOR;
			}
			else {
				box_color = Config::BOX_COLOR;
				text_color = Config::TEXT_COLOR;
			}
			Util::plotBox(img, b[k].x_min, b[k].y_min,
						  b[k].x_max, b[k].y_max,
						  box_color, Config::BOX_LINE_WIDTH);
			std::stringstream text;
			text << Config::POST_TEXT[b[k].class_id] << ": " << 100 * b[k].score << "%";

			cv::putText(img, text.str(),
						cv::Point(b[k].x_min, b[k].y_min-Config::TEXT_FONT_SIZE-10),
						cv::FONT_HERSHEY_PLAIN, Config::TEXT_FONT_SIZE,
						cv::Scalar(text_color[0], text_color[1], text_color[2]),
						(int)Config::TEXT_LINE_WIDTH);
			if(b[k].class_id==Config::TARGET_CLASS)alarm = 1;
		}

	}
}

void Postprocessor::Init()
{
	if (!m_ops) {
		m_ops = createSharedRef<Factory<PostprocessorOps>>();
	}
	m_ops->registerType<HelmetDetectionPost>(Config::POSTPROCESS_NAME);
	m_worker = m_ops->create(Config::POSTPROCESS_NAME);
}

void Postprocessor::Run(const SharedRef<TrtResults> &res, cv::Mat &img,int &alarm)
{
	if (!INIT_FLAG) {
		Init();
		INIT_FLAG = true;
	}
	m_worker->Run(res, img,alarm);
}

Postprocessor::~Postprocessor()
{
	if (m_ops) {
		m_ops->destroy();
	}
}

}