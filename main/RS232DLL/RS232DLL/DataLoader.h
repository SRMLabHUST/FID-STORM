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
	// 【1】初始化所有变量，分配内存
	bool init(string fileName, int batchSize, std::mutex* pMutex,std::condition_variable* pCondVal,bool* pIsArrFull,bool* isProcessOver,bool* pIsDataTakeOff,bool fp16);

	// 【2】取出所有的块，然后读取文件
	bool imgRead();

	// 【3】归一化
	bool imageNormalize(int curFrame);

	// 【3】去初始化
	bool deInit();

public:
	string fileName;			// 文件名
	int batchSize;				// batch图像

	bool fp16;

	float * imgRaw;				// batch图像内存空间, fp32
	half* imgRaw_fp16;			// batch图像内存空间，fp16

	unsigned short * imgTemp;	// 读取的图像，临时存放
	int imgMaxV = 0, imgMinV = 1000000;	// 图像的最大值，最小值

	int imageTotal;				// 总的图像帧数
	int curFrame = 0;			// 当前frame
	int curBatch = 0;			// 当前batch
	int imageWidth;				// 图像宽度
	int imageHeight;			// 图像高度

private:
	void imgMaxMinFind();			// 查找图像最大值，最小值

public:
	// 互斥量（不允许拷贝构造）
	std::mutex * pMutex = nullptr;
	std::condition_variable * pCondVal = nullptr;
	bool* pIsArrFull = nullptr;
	bool* pIsProcessOver = nullptr;
	bool* pIsDataTakeOff = nullptr;
};