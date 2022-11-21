#pragma once
#include <iostream>

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"

#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <malloc.h>
#include"tinytiffreader.h"
#include"tinytiffwriter.h"

using namespace half_float;
using samplesCommon::SampleUniquePtr;

using namespace std;

class DataLoader
{
public:
	// ��1����ʼ�����б����������ڴ�
	bool init(string fileName, int batchSize, std::mutex* pMutex,std::condition_variable* pCondVal,bool* pIsArrFull,bool* isProcessOver,bool* pIsDataTakeOff,bool fp16);

	// ��2��ȡ�����еĿ飬Ȼ���ȡ�ļ�
	bool imgRead();

	// ��3����һ��
	bool imageNormalize(int curFrame);

	// ��3��ȥ��ʼ��
	bool deInit();

public:
	string fileName;			// �ļ���
	int batchSize;				// batchͼ��

	bool fp16;

	float * imgRaw;				// batchͼ���ڴ�ռ�, fp32
	half* imgRaw_fp16;			// batchͼ���ڴ�ռ䣬fp16

	unsigned short * imgTemp;	// ��ȡ��ͼ����ʱ���
	int imgMaxV = 0, imgMinV = 1000000;	// ͼ������ֵ����Сֵ

	int imageTotal;				// �ܵ�ͼ��֡��
	int curFrame = 0;			// ��ǰframe
	int curBatch = 0;			// ��ǰbatch
	int imageWidth;				// ͼ����
	int imageHeight;			// ͼ��߶�

private:
	void imgMaxMinFind();			// ����ͼ�����ֵ����Сֵ

public:
	// �������������������죩
	std::mutex * pMutex = nullptr;
	std::condition_variable * pCondVal = nullptr;
	bool* pIsArrFull = nullptr;
	bool* pIsProcessOver = nullptr;
	bool* pIsDataTakeOff = nullptr;
};