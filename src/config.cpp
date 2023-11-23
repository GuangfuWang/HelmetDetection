#include <fstream>
#include "config.h"
#include "macro.h"
#include "util.h"

namespace helmet
{



void Config::LoadConfigFile(int argc, char **argv, const std::string &file)
{
	if (init)return;
	init = true;
	std::cout << "Start parsing the config file for Helmet Detection" << std::endl;
	MODEL_NAME = DEPLOY_MODEL;
	std::cout << "Read from cmake config with model name: " << DEPLOY_MODEL << std::endl;
	INPUT_NAME = parseNames(MODEL_INPUT_NAME, ' ');
	std::cout << "Read from cmake config with inputs: [" << MODEL_INPUT_NAME <<"]"<< std::endl;
	OUTPUT_NAMES.clear();
	OUTPUT_NAMES = parseNames(MODEL_OUTPUT_NAMES, ' ');
	std::cout << "Read from cmake config with outputs: [" << MODEL_OUTPUT_NAMES <<"]"<< std::endl;

	if (!checkFileExist(file)) {
		std::cerr << "Config file non exists! Aborting..." << std::endl;
	}

	YAML::Node config;
//	config = YAML::LoadFile(file);
	std::ifstream ai_model(file, std::ios::in);
	if (!ai_model) {
		return;
	}
	std::stringstream m_str;
	m_str << ai_model.rdbuf();
	ai_model.close();
	config = YAML::Load(m_str.str());

	if (config["MODEL"].IsDefined()) {
		auto model_node = config["MODEL"];
		if (model_node["MODEL_NAME"].IsDefined()) {
			MODEL_NAME = model_node["MODEL_NAME"].as<std::string>();
			std::cout << "Read from YAML with model name: " << MODEL_NAME << std::endl;
		}
		if (model_node["BACKBONE"].IsDefined()) {
			BACKBONE = model_node["BACKBONE"].as<std::string>();
			std::cout << "Read from YAML with backbone: " << BACKBONE << std::endl;
		}
		if (model_node["INPUT_NAME"].IsDefined()) {
			INPUT_NAME.clear();
			INPUT_NAME = model_node["INPUT_NAME"].as<std::vector<std::string>>();

			std::cout << "Read from YAML with inputs: [\t";
			for (auto &each : INPUT_NAME)
				std::cout << each << "\t";
			std::cout <<"]"<< std::endl;
		}
		if (model_node["OUTPUT_NAMES"].IsDefined()) {
			OUTPUT_NAMES.clear();
			OUTPUT_NAMES = model_node["OUTPUT_NAMES"].as<std::vector<std::string>>();

			std::cout << "Read from YAML with outputs: [\t";
			for (auto &each : OUTPUT_NAMES)
				std::cout << each << "\t";
			std::cout <<"]"<< std::endl;
		}
	}
	else {
		std::cerr << "Please set MODEL, " << std::endl;
	}

	if (config["DATA"].IsDefined()) {
		auto model_node = config["DATA"];
		if (model_node["VIDEO_NAME"].IsDefined()) {
			VIDEO_FILE = model_node["VIDEO_NAME"].as<std::string>();
			std::cout << "Read from YAML with videos: " << VIDEO_FILE << std::endl;
		}
		if (model_node["RTSP_SITE"].IsDefined()) {
			RTSP_SITE = model_node["RTSP_SITE"].as<std::string>();
			std::cout << "Read from YAML with backbone: " << RTSP_SITE << std::endl;
		}
		if (model_node["INPUT_SHAPE"].IsDefined()) {
			INPUT_SHAPE = model_node["INPUT_SHAPE"].as<std::vector<int>>();

			std::cout << "Read from YAML with input shapes: [\t";
			for (auto &each : INPUT_SHAPE)
				std::cout << each << "\t";
			std::cout <<"]"<< std::endl;
		}
	}
	else {
		std::cerr << "Please set DATA..." << std::endl;
	}

	if (config["PIPELINE"].IsDefined()) {
		auto model_node = config["PIPELINE"];
		if (model_node["STRIDE"].IsDefined()) {
			STRIDE = model_node["STRIDE"].as<unsigned int>();
			std::cout << "Read from YAML with stride: " << STRIDE << std::endl;
		}
		if (model_node["INTERP"].IsDefined()) {
			INTERP = model_node["INTERP"].as<unsigned int>();
			std::cout << "Read from YAML with interpolation: " << INTERP << std::endl;
		}
		if (model_node["SAMPLE_INTERVAL"].IsDefined()) {
			SAMPLE_INTERVAL = model_node["SAMPLE_INTERVAL"].as<unsigned int>();
			std::cout << "Read from YAML with sample interval: " << SAMPLE_INTERVAL << std::endl;
		}
		if (model_node["TRIGGER_LEN"].IsDefined()) {
			TRIGGER_LEN = model_node["TRIGGER_LEN"].as<unsigned int>();
			std::cout << "Read from YAML with trigger length: " << TRIGGER_LEN << std::endl;
		}
		if (model_node["BATCH_SIZE"].IsDefined()) {
			BATCH_SIZE = model_node["BATCH_SIZE"].as<unsigned int>();
			std::cout << "Read from YAML with batch size: " << BATCH_SIZE << std::endl;
		}
		if (model_node["THRESHOLD"].IsDefined()) {
			THRESHOLD = model_node["THRESHOLD"].as<float>();
			std::cout << "Read from YAML with threshold: " << THRESHOLD << std::endl;
		}
		if (model_node["SCORE_THRESHOLD"].IsDefined()) {
			SCORE_THRESHOLD = model_node["SCORE_THRESHOLD"].as<float>();
			std::cout << "Read from YAML with score threshold: " << SCORE_THRESHOLD << std::endl;
		}
		if (model_node["TARGET_CLASS"].IsDefined()) {
			TARGET_CLASS = model_node["TARGET_CLASS"].as<unsigned int>();
			std::cout << "Read from YAML with target class: " << TARGET_CLASS << std::endl;
		}
		if (model_node["ENABLE_SCALE"].IsDefined()) {
			ENABLE_SCALE = model_node["ENABLE_SCALE"].as<bool>();
			std::cout << "Read from YAML with enable scale: " << ENABLE_SCALE << std::endl;
		}
		if (model_node["KEEP_RATIO"].IsDefined()) {
			KEEP_RATIO = model_node["KEEP_RATIO"].as<bool>();
			std::cout << "Read from YAML with keep ratio option: " << KEEP_RATIO << std::endl;
		}
		if (model_node["TIMING"].IsDefined()) {
			TIMING = model_node["TIMING"].as<bool>();
			std::cout << "Read from YAML with timing: " << TIMING << std::endl;
		}
		if (model_node["TARGET_SIZE"].IsDefined()) {
			TARGET_SIZE = model_node["TARGET_SIZE"].as<std::vector<int>>();

			std::cout << "Read from YAML with target size: [\t";
			for (auto &each : TARGET_SIZE)
				std::cout << each << "\t";
			std::cout <<"]"<< std::endl;
		}
		if (model_node["TRAIN_SIZE"].IsDefined()) {
			TRAIN_SIZE = model_node["TRAIN_SIZE"].as<std::vector<int>>();

			std::cout << "Read from YAML with train size: [\t";
			for (auto &each : TRAIN_SIZE)
				std::cout << each << "\t";
			std::cout <<"]"<< std::endl;
		}
		if (model_node["SHORT_SIZE"].IsDefined()) {
			SHORT_SIZE = model_node["SHORT_SIZE"].as<unsigned int>();
			std::cout << "Read from YAML with short size: " << SHORT_SIZE << std::endl;
		}
		if (model_node["PIPELINE_TYPE"].IsDefined()) {
			PIPELINE_TYPE = model_node["PIPELINE_TYPE"].as<std::vector<std::string>>();

			std::cout << "Read from YAML with pipelines: [\t";
			for (auto &each : PIPELINE_TYPE)
				std::cout << each << "\t";
			std::cout <<"]"<< std::endl;
		}
		if (model_node["N_MEAN"].IsDefined()) {
			N_MEAN = model_node["N_MEAN"].as<std::vector<float>>();

			std::cout << "Read from YAML with mean: [\t";
			for (auto &each : N_MEAN)
				std::cout << each << "\t";
			std::cout <<"]"<< std::endl;
		}
		if (model_node["N_STD"].IsDefined()) {
			N_STD = model_node["N_STD"].as<std::vector<float>>();

			std::cout << "Read from YAML with std: [\t";
			for (auto &each : N_STD)
				std::cout << each << "\t";
			std::cout <<"]"<< std::endl;
		}

	}
	else {
		std::cerr << "Please set PIPELINE, " << std::endl;
	}

	if (config["POSTPROCESS"].IsDefined()) {
		auto model_node = config["POSTPROCESS"];
		if (model_node["POST_MODE"].IsDefined()) {
			POST_MODE = model_node["POST_MODE"].as<int>();
			std::cout << "Read from YAML with post mode: " << POST_MODE << std::endl;
		}
		if (model_node["TEXT_COLOR"].IsDefined()) {
			TEXT_COLOR = model_node["TEXT_COLOR"].as<std::vector<unsigned char>>();
			std::swap(TEXT_COLOR[0], TEXT_COLOR[2]);

			std::cout << "Read from YAML with text color: [\t";
			for (auto &each : TEXT_COLOR)
				std::cout << (int)each << "\t";
			std::cout <<"]"<< std::endl;
		}
		if (model_node["BOX_COLOR"].IsDefined()) {
			BOX_COLOR = model_node["BOX_COLOR"].as<std::vector<unsigned char>>();
			std::swap(BOX_COLOR[0], BOX_COLOR[2]);

			std::cout << "Read from YAML with box colors: [\t";
			for (auto &each : BOX_COLOR)
				std::cout << (int)each << "\t";
			std::cout <<"]"<< std::endl;
		}
		if (model_node["ALARM_TEXT_COLOR"].IsDefined()) {
			ALARM_TEXT_COLOR = model_node["ALARM_TEXT_COLOR"].as<std::vector<unsigned char>>();
			std::swap(ALARM_TEXT_COLOR[0], ALARM_TEXT_COLOR[2]);

			std::cout << "Read from YAML with alarm text: [\t";
			for (auto &each : ALARM_TEXT_COLOR)
				std::cout << (int)each << "\t";
			std::cout <<"]"<< std::endl;
		}
		if (model_node["ALARM_BOX_COLOR"].IsDefined()) {
			ALARM_BOX_COLOR = model_node["ALARM_BOX_COLOR"].as<std::vector<unsigned char>>();
			std::swap(ALARM_BOX_COLOR[0], ALARM_BOX_COLOR[2]);

			std::cout << "Read from YAML with alarm box color: [\t";
			for (auto &each : ALARM_BOX_COLOR)
				std::cout << (int)each << "\t";
			std::cout <<"]"<< std::endl;
		}
		if (model_node["TEXT_LINE_WIDTH"].IsDefined()) {
			TEXT_LINE_WIDTH = model_node["TEXT_LINE_WIDTH"].as<float>();
			std::cout << "Read from YAML with text line width: " << TEXT_LINE_WIDTH << std::endl;
		}
		if (model_node["BOX_LINE_WIDTH"].IsDefined()) {
			BOX_LINE_WIDTH = model_node["BOX_LINE_WIDTH"].as<int>();
			std::cout << "Read from YAML with box line width: " << BOX_LINE_WIDTH << std::endl;
		}
		if (model_node["TEXT_FONT_SIZE"].IsDefined()) {
			TEXT_FONT_SIZE = model_node["TEXT_FONT_SIZE"].as<float>();
			std::cout << "Read from YAML with text font size: " << TEXT_FONT_SIZE << std::endl;
		}
		if (model_node["TEXT_OFF_X"].IsDefined()) {
			TEXT_OFF_X = model_node["TEXT_OFF_X"].as<int>();
			if (TEXT_OFF_X < 0) {
				TEXT_OFF_X = INPUT_SHAPE.back() / 2 - 5;
			}
			std::cout << "Read from YAML with text pos offset x: " << TEXT_OFF_X << std::endl;
		}
		if (model_node["TEXT_OFF_Y"].IsDefined()) {
			TEXT_OFF_Y = model_node["TEXT_OFF_Y"].as<int>();
			std::cout << "Read from YAML with text pos offset y: " << TEXT_OFF_Y << std::endl;
		}
		if (model_node["ALARM_COUNT"].IsDefined()) {
			ALARM_COUNT = model_node["ALARM_COUNT"].as<int>();
			std::cout << "Read from YAML with ALARM_COUNT: " << ALARM_COUNT << std::endl;
		}
		if (model_node["POSTPROCESS_NAME"].IsDefined()) {
			POSTPROCESS_NAME = model_node["POSTPROCESS_NAME"].as<std::string>();
			std::cout << "Read from YAML with post process name: " << POSTPROCESS_NAME << std::endl;
		}
		if (model_node["POST_TEXT"].IsDefined()) {
			POST_TEXT = model_node["POST_TEXT"].as<std::vector<std::string>>();

			std::cout << "Read from YAML with post text: [\t";
			for (auto &each : POST_TEXT)
				std::cout << each << "\t";
			std::cout <<"]"<< std::endl;
		}
		if (model_node["POST_TEXT_FONT_FILE"].IsDefined()) {
			POST_TEXT_FONT_FILE = model_node["POST_TEXT_FONT_FILE"].as<std::string>();
			std::cout << "Read from YAML with post text fonts: "<<POST_TEXT_FONT_FILE<<std::endl;
		}
	}
	else {
		std::cerr << "Please set MODEL, " << std::endl;
	}

	if (argc < 2)return;

	cmdline::parser parser;
	parser.add<std::string>("input_name", 'i', "Input layer name for trt.", false);
	parser.add<std::string>("output_names", 'o', "Output layer names for trt.", false);
	parser.add<std::string>("model_name", 'm', "Model name for trt.", false);
	parser.add<std::string>("video_file", 'v', "Video file for trt.", false);
	parser.parse_check(argc, argv);

	std::string InLayerName = parser.get<std::string>("input_name");
	std::string OutLayerNames = parser.get<std::string>("output_names");
	std::string ModelName = parser.get<std::string>("model_name");
	std::string VideoFile = parser.get<std::string>("video_file");

	if (!InLayerName.empty()) {
		INPUT_NAME = parseNames(InLayerName, ' ');

		std::cout << "Read from cmd with inputs: [\t";
		for (auto &each : INPUT_NAME) {
			std::cout << each << "\t";
		}
		std::cout <<"]"<< std::endl;
	}
	if (!OutLayerNames.empty()) {
		OUTPUT_NAMES.clear();
		OUTPUT_NAMES = parseNames(OutLayerNames, ' ');

		std::cout << "Read from cmd with outputs: [\t";
		for (auto &each : OUTPUT_NAMES) {
			std::cout << each << "\t";
		}
		std::cout <<"]"<< std::endl;
	}
	if (!ModelName.empty()) {
		MODEL_NAME = ModelName;
		std::cout<<"Read from cmd with model file: "<<MODEL_NAME<<std::endl;
	}

	if (!helmet::checkFileExist(MODEL_NAME)) {
		std::cout << MODEL_NAME << std::endl;
		std::cerr << "Model does not exists!" << std::endl;
		std::cerr << "Please check the model path..." << std::endl;
	}

	if (!VideoFile.empty()) {
		VIDEO_FILE = VideoFile;
		std::cout<<"Read from cmd with video file: "<<VIDEO_FILE<<std::endl;
	}
}

Config::Config(int argc, char **argv, const std::string &file)
{
	LoadConfigFile(argc,argv,file);
}

}

