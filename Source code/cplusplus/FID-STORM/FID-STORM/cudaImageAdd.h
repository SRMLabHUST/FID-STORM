#pragma once
#include<cuda_runtime.h>

class cudaImageAdd
{
public:
	int memorySize;		// Data size
	int imgWidth  = 2048;
	int imgHeight = 2048;
	int batchSize = 13;

	// Input data
	float* renderImg_host;
	float* renderImg_device;

	// Output data
	float* renderImgOut_device;
	float* renderImgOut_host;

	bool init();
	bool deinit();

	bool render();
};

__global__ void imgAdd(float* renderImg_device, float* renderImgOut_device, int imgWidth, int imgHeight, int batchSize, bool fp16);
void imgAddTop(float* renderImg_device, float* renderImgOut_device, int imgWidth, int imgHeight, int batchSize,bool fp16);