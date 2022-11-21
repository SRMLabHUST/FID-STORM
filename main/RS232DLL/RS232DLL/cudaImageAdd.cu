#include"cudaImageAdd.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
#include <iostream>
using namespace std;

//bool cudaImageAdd::init()
//{
//	// ��������ͼ�������ڴ桢�豸�ڴ�
//	this->memorySize = imgHeight * imgWidth * batchSize;
//	this->renderImg_host  = new float[memorySize];
//	cudaMalloc((void**)&this->renderImg_device, memorySize*sizeof(float));
//
//	// ��ʼ�������ڴ�
//	//memset(this->renderImg_host, 0, memorySize*sizeof(float));
//	for (int i = 0; i < imgHeight*imgWidth*batchSize;i++)
//	{
//		renderImg_host[i] = 1;
//	}
//
//	// ������ͼ��������ֵ�������豸
//	cudaError_t state = cudaMemcpy(this->renderImg_device, this->renderImg_host, this->memorySize * sizeof(float), cudaMemcpyHostToDevice);
//	cout << "����ͼ���Ƿ����" << state << endl;
//
//	//float* test = new float[2048 * 2048 * 13];
//	//cudaMemcpy(test, renderImg_device, memorySize * sizeof(float), cudaMemcpyDeviceToHost);
//
//	// �������ͼ�������ڴ棬�豸�ڴ�
//	this->renderImgOut_host = new float[imgHeight*imgWidth];
//	cudaMalloc((void**)&this->renderImgOut_device, imgHeight*imgWidth * sizeof(float));
//
//
//	return true;
//}
//
//bool cudaImageAdd::deinit()
//{
//	// �ͷ������ڴ�
//	delete[] this->renderImg_host;
//	cudaError state = cudaFree(this->renderImg_device);
//	cout << "����ͼ���豸�ڴ��ͷ��Ƿ����" << state << endl;
//
//	delete[] this->renderImgOut_host;
//	state =  cudaFree(this->renderImgOut_device);
//	cout << "���ͼ���豸�ڴ��ͷ��Ƿ����" << state << endl;
//	return true; 
//}
//
//bool cudaImageAdd::render() 
//{
//	//��2�� ���ݵļӼ�
//	int threadPerBlock = 256;
//	int blockNums = (imgHeight * imgWidth + threadPerBlock  -1)/ threadPerBlock;
//
//	for(int i = 0;i<10;i++)
//		imgAdd <<<blockNums, threadPerBlock >>> (renderImg_device, renderImgOut_device, imgWidth, imgHeight, batchSize);
//
//	//��3�� ���������ݿ���������
//	cudaError_t state = cudaMemcpy(this->renderImgOut_host, this->renderImgOut_device, imgHeight * imgWidth * sizeof(float), cudaMemcpyDeviceToHost);
//
//	cout << "���ͼ���Ƿ����" << state << endl;
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