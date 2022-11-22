#include"cudaImageAdd.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
#include <iostream>
using namespace std;

//bool cudaImageAdd::init()
//{
//	¡¾1¡¿ Build input image host memory and device memory
//	this->memorySize = imgHeight * imgWidth * batchSize;
//	this->renderImg_host  = new float[memorySize];
//	cudaMalloc((void**)&this->renderImg_device, memorySize*sizeof(float));
//
//	****Initialize the host memory
//	//memset(this->renderImg_host, 0, memorySize*sizeof(float));
//	for (int i = 0; i < imgHeight*imgWidth*batchSize;i++)
//	{
//		renderImg_host[i] = 1;
//	}
//
//	****Copy the input image host value to the device
//	cudaError_t state = cudaMemcpy(this->renderImg_device, this->renderImg_host, this->memorySize * sizeof(float), cudaMemcpyHostToDevice);
//	cout << "Input image error£º" << state << endl;
//
//	//float* test = new float[2048 * 2048 * 13];
//	//cudaMemcpy(test, renderImg_device, memorySize * sizeof(float), cudaMemcpyDeviceToHost);
//
//	****Build output image host memory, device memory
//	this->renderImgOut_host = new float[imgHeight*imgWidth];
//	cudaMalloc((void**)&this->renderImgOut_device, imgHeight*imgWidth * sizeof(float));
//
//	return true;
//}
//
//bool cudaImageAdd::deinit()
//{
//	****Free host memory
//	delete[] this->renderImg_host;
//	cudaError state = cudaFree(this->renderImg_device);
//	cout << "Input image device memory release error£º" << state << endl;
//
//	delete[] this->renderImgOut_host;
//	state =  cudaFree(this->renderImgOut_device);
//	cout << "Output image device memory release error£º" << state << endl;
//	return true; 
//}
//
//bool cudaImageAdd::render() 
//{
//	¡¾2¡¿ Addition and subtraction of data
//	int threadPerBlock = 256;
//	int blockNums = (imgHeight * imgWidth + threadPerBlock  -1)/ threadPerBlock;
//
//	for(int i = 0;i<10;i++)
//		imgAdd <<<blockNums, threadPerBlock >>> (renderImg_device, renderImgOut_device, imgWidth, imgHeight, batchSize);
//
//	¡¾3¡¿ Copy the calculated data to the host
//	cudaError_t state = cudaMemcpy(this->renderImgOut_host, this->renderImgOut_device, imgHeight * imgWidth * sizeof(float), cudaMemcpyDeviceToHost);
//
//	cout << "Output image error£º" << state << endl;
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