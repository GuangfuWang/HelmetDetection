#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include "cmdline.h"

namespace helmet
{
/**
 * @brief config class for deployment. should not be derived.
 */
class Config final
{
public:
	explicit Config(int argc, char **argv, const std::string &file);
	/**
	 * @brief loading the config yaml file, default folder is ./config/helmet_detection.yaml
	 * @param argc terminal arg number.
	 * @param argv terminal arg values.
	 * @param file config file full path.
	 * @note priority: 1 Terminal; 2 Config file; 3 Compilation settings
	 */
private:
	void LoadConfigFile(int argc, char **argv, const std::string &file);

public:
	std::string MODEL_NAME = "../models/helmet_model.engine";
	std::string BACKBONE = "ResNet50";
	std::string VIDEO_FILE;
	std::string RTSP_SITE = "/url/to/rtsp/site";
	std::vector<int> INPUT_SHAPE = {1, 8, 3, 320, 320};
	std::vector<std::string> INPUT_NAME = {"im_shape", "image", "scale_factor"};
	std::vector<std::string> OUTPUT_NAMES = {"dets", "num_dets"};

	unsigned int STRIDE = 2;
	unsigned int INTERP = 0;
	unsigned int SAMPLE_INTERVAL = 1;
	unsigned int TRIGGER_LEN = 1;
	unsigned int BATCH_SIZE = 1;
	float THRESHOLD = 0.8f;
	float SCORE_THRESHOLD = 0.6f;
	unsigned int TARGET_CLASS = 1;
	std::vector<int> TARGET_SIZE = {608, 608};
	std::vector<int> TRAIN_SIZE = {608, 608};
	unsigned int SHORT_SIZE = 340;
	std::vector<std::string> PIPELINE_TYPE =
		{"TopDownEvalAffine", "Resize", "LetterBoxResize", "NormalizeImage"};
	int SAMPLE_DATA = 3;

	std::vector<float> N_MEAN = {0.485f, 0.456f, 0.406f};
	std::vector<float> N_STD = {0.229f, 0.224f, 0.225f};
	bool ENABLE_SCALE = true;
	bool KEEP_RATIO = true;
	bool TIMING = true;
	int POST_MODE = 0;
	std::vector<unsigned char> TEXT_COLOR = {0, 0, 255};
	std::vector<unsigned char> BOX_COLOR = {0, 0, 255};
	std::vector<unsigned char> ALARM_TEXT_COLOR = {255, 0, 0};
	std::vector<unsigned char> ALARM_BOX_COLOR = {255, 0, 0};
	float TEXT_LINE_WIDTH = 2.0f;
	int BOX_LINE_WIDTH = 2.0;
	float TEXT_FONT_SIZE = 1.8f;
	int TEXT_OFF_X = 450;
	int TEXT_OFF_Y = 50;
	int ALARM_COUNT = 5;
	std::string POSTPROCESS_NAME = "HelmetDetectionPost";
	std::vector<std::string> POST_TEXT = {"未佩戴安全帽", "佩戴安全帽"};
	std::string POST_TEXT_FONT_FILE = "";
	bool init = false;
};
}