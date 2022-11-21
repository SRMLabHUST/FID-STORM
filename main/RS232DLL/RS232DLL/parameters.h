#pragma once
#include<argsParser.h>
#include<half.h>
//#include "cuda_fp16.h"
//
using namespace half_float;

struct Params : public samplesCommon::OnnxSampleParams, samplesCommon::Args
{
public:
	// ���� zzw
	static const int fileNums = 13;					// file numbers ,.tif
	int batchSize = fileNums;
	std::string inputFileName[fileNums];			// <input fileName,.tif>
	std::string inputNormFileName[fileNums];		// <input Norm fileName, .tif>
	std::string outputFileName[fileNums];			// <output fileName,.tif>
	int inputWidth, inputHight;
	int scaleFactor;
	int outputWidth = inputWidth * scaleFactor;
	int outputHight = inputHight * scaleFactor;
	string engine_file;

	string inputDataDir ;	// ����������ݵ��ļ���
	string outputDataDir;	// ���������ݵ��ļ���

	// �������
	float* renderImgOut_device;
	float* renderImgOut_host;

	half* renderImgOut_device_fp16;
	half* renderImgOut_host_fp16;

	//std::string fileDir = "";						// file directorys;
	std::string fileDir;							// file directory;
	std::string outputTiffName;
};