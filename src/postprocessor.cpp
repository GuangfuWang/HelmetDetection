#include <opencv2/imgproc.hpp>
#include <opencv2/freetype.hpp>
#include "postprocessor.h"
#include "config.h"
#include <cmath>

namespace helmet
{

void HelmetDetectionPost::Run(const SharedRef<TrtResults> &res, cv::Mat &img,int &alarm)
{
	//our simple program will only draw letters on top of images.
//	auto flag = static_cast<PostProcessFlag>(m_config->POST_MODE);
//	assert(flag == PostProcessFlag::DRAW_BOX_LETTER);
	alarm = 0;

	std::vector<float> dets;
	std::vector<float> num_dets;
	res->Get(m_config->OUTPUT_NAMES[0], dets);
	res->Get(m_config->OUTPUT_NAMES[1], num_dets);

	float scale_x = (float)img.cols / (float)m_config->TARGET_SIZE[1];
	float scale_y = (float)img.rows / (float)m_config->TARGET_SIZE[0];

	std::vector<Box> curr;
	for (int j = 0; j < 100; ++j) {
		Box b;
		b.class_id = round2int(dets[0+j*6]);
		b.score = dets[1+j*6];
		b.x_min = (int)(dets[2+j*6] * scale_x);
		b.y_min = (int)(dets[3+j*6] * scale_y);
		b.x_max = (int)(dets[4+j*6] * scale_x);
		b.y_max = (int)(dets[5+j*6] * scale_y);
		curr.push_back(b);
	}
	auto b = curr;
	bool ff = false;
	///@note the putText method does not have GPU version since it quite slow running on GPU for per pixel ops.
	for (int k = 0; k < b.size(); ++k) {
		if(b[k].class_id>1)continue;
		if (b[k].score > m_config->SCORE_THRESHOLD) {
			std::vector<unsigned char> box_color;
			box_color.resize(3);
			std::vector<unsigned char> text_color;
			box_color.resize(3);
			if(b[k].class_id==m_config->TARGET_CLASS){
				box_color = m_config->ALARM_BOX_COLOR;
				text_color = m_config->ALARM_TEXT_COLOR;
			}
			else {
				box_color = m_config->BOX_COLOR;
				text_color = m_config->TEXT_COLOR;
			}
			plotBox(img, b[k].x_min, b[k].y_min,
						  b[k].x_max, b[k].y_max,
						  box_color, m_config->BOX_LINE_WIDTH);
			std::stringstream text;
			float percent = 100 * b[k].score;
//			text << m_config->POST_TEXT[b[k].class_id] << ": " << std::fixed<<std::setprecision(3)<<percent << "%";
			const int line_type = 8;
			m_font->putText(img,m_config->POST_TEXT[b[k].class_id],cv::Point(b[k].x_min, b[k].y_min-m_config->TEXT_FONT_SIZE-10),
							m_config->TEXT_FONT_SIZE,cv::Scalar(text_color[0], text_color[1], text_color[2]),
							(int)m_config->TEXT_LINE_WIDTH,line_type,false);
//			m_font->putText(img, text.str(),
//						cv::Point(b[k].x_min, b[k].y_min-m_config->TEXT_FONT_SIZE-10),
//						cv::FONT_HERSHEY_PLAIN, m_config->TEXT_FONT_SIZE,
//						cv::Scalar(text_color[0], text_color[1], text_color[2]),
//						(int)m_config->TEXT_LINE_WIDTH);
			if(b[k].class_id==m_config->TARGET_CLASS){
				m_latency+=2;
				if(m_latency>2*m_config->ALARM_COUNT){
					alarm = 1;
					m_latency = 0;
				}
				ff = true;
			}
		}
	}
	if(!ff){
		m_latency--;
	}
	if(m_latency<0)m_latency=0;
}

void Postprocessor::Init()
{
//	if (!m_ops) {
//		m_ops = createSharedRef<Factory<PostprocessorOps>>(m_config);
//	}
//	m_ops->registerType<HelmetDetectionPost>(m_config->POSTPROCESS_NAME);
	m_worker = new HelmetDetectionPost(m_config);
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
//	if (m_ops) {
//		m_ops->destroy();
//	}
	if(m_worker)delete m_worker;
}

}
