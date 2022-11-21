#include"cudaImageAdd.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
#include <iostream>
using namespace std;

//bool cudaImageAdd::init()
//{
//	// 构建输入图像主机内存、设备内存
//	this->memorySize = imgHeight * imgWidth * batchSize;
//	this->renderImg_host  = new float[memorySize];
//	cudaMalloc((void**)&this->renderImg_device, memorySize*sizeof(float));
//
//	// 初始化主机内存
//	//memset(this->renderImg_host, 0, memorySize*sizeof(float));
//	for (int i = 0; i < imgHeight*imgWidth*batchSize;i++)
//	{
//		renderImg_host[i] = 1;
//	}
//
//	// 将输入图像主机数值拷贝到设备
//	cudaError_t state = cudaMemcpy(this->renderImg_device, this->renderImg_host, this->memorySize * sizeof(float), cudaMemcpyHostToDevice);
//	cout << "输入图像是否出错：" << state << endl;
//
//	//float* test = new float[2048 * 2048 * 13];
//	//cudaMemcpy(test, renderImg_device, memorySize * sizeof(float), cudaMemcpyDeviceToHost);
//
//	// 构建输出图像主机内存，设备内存
//	this->renderImgOut_host = new float[imgHeight*imgWidth];
//	cudaMalloc((void**)&this->renderImgOut_device, imgHeight*imgWidth * sizeof(float));
//
//
//	return true;
//}
//
//bool cudaImageAdd::deinit()
//{
//	// 释放主机内存
//	delete[] this->renderImg_host;
//	cudaError state = cudaFree(this->renderImg_device);
//	cout << "输入图像设备内存释放是否出错：" << state << endl;
//
//	delete[] this->renderImgOut_host;
//	state =  cudaFree(this->renderImgOut_device);
//	cout << "输出图像设备内存释放是否出错：" << state << endl;
//	return true; 
//}
//
//bool cudaImageAdd::render() 
//{
//	//【2】 数据的加减
//	int threadPerBlock = 256;
//	int blockNums = (imgHeight * imgWidth + threadPerBlock  -1)/ threadPerBlock;
//
//	for(int i = 0;i<10;i++)
//		imgAdd <<<blockNums, threadPerBlock >>> (renderImg_device, renderImgOut_device, imgWidth, imgHeight, batchSize);
//
//	//【3】 计算后的数据拷贝到主机
//	cudaError_t state = cudaMemcpy(this->renderImgOut_host, this->renderImgOut_device, imgHeight * imgWidth * sizeof(float), cudaMemcpyDeviceToHost);
//
//	cout << "输出图像是否出错：" << state << endl;
//	printf("pause");
//	return true;
//}

__global__ void imgAdd(float* renderImg_device, float* renderImgOut_device, int imgWidth, int imgHeight, int batchSize, bool fp16)
{
	int blockId = blockIdx.x;
	int id = blockId * blockDim.x + threadIdx.x;

	if (id < imgWidth * imgHeight)
	{
		for (int i = 0; i < batchSize; i++)
		{
			if (!fp16)
				renderImgOut_device[id] += renderImg_device[id + i * imgWidth*imgHeight];
			else
				renderImgOut_device[id] += __half2float(renderImgOut_device[id + i * imgWidth*imgHeight]);		// fp16
		}
	}
}

void imgAddTop(float* renderImg_device, float* renderImgOut_device, int imgWidth, int imgHeight, int batchSize,bool fp16)
{
	int threadPerBlock = 256;
	int blockNums = (imgHeight * imgWidth + threadPerBlock - 1) / threadPerBlock;
	imgAdd << <blockNums, threadPerBlock >> > (renderImg_device, renderImgOut_device, imgWidth, imgHeight, batchSize,fp16);
}



//int main_cudaImageAdd00()
//{
//	cudaImageAdd add;
//	add.init();
//	add.render();
//	add.deinit();
//	return 0;
//}