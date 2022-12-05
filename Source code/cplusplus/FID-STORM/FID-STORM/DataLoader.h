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
	// 【1】Initialize all variables and allocate memory
	bool init(string fileName, int batchSize, std::mutex* pMutex,std::condition_variable* pCondVal,bool* pIsArrFull,bool* isProcessOver,bool* pIsDataTakeOff,bool fp16);

	// 【2】Fetch all the blocks and then read the file
	bool imgRead();

	// 【3】Normalization
	bool imageNormalize(int curFrame);

	// 【3】De-initialization
	bool deInit();

public:
	string fileName;			
	int batchSize;				// batch image

	bool fp16;

	float * imgRaw;				// batch Image memory space, fp32
	half* imgRaw_fp16;			// batch Image memory space，fp16

	unsigned short * imgTemp;	// Read the image, temporary storage
	int imgMaxV = 0, imgMinV = 1000000;	// The maximum value of the image, the minimum value

	int imageTotal;				// Total image frames
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