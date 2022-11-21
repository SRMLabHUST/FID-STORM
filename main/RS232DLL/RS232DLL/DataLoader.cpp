#include <iostream>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <malloc.h>
#include"tinytiffreader.h"
#include"tinytiffwriter.h"
#include"DataLoader.h"

using namespace half_float;
using namespace std;

/****************************DataLoader的函数实现*********************************************/
bool DataLoader::init(string fileName, int batchSize, std::mutex* pMutex, std::condition_variable* pCondVal,bool* pIsArrFull,bool * pIsProcessOver, bool* pIsDataTakeOff,bool fp16)
{
	this->fileName  = fileName;
	this->batchSize = batchSize;

	this->pMutex			= pMutex;			// 互斥量（不允许拷贝构造）
	this->pCondVal			= pCondVal;			// 条件变量
	this->pIsArrFull		= pIsArrFull;		// 数组是否已满
	this->pIsProcessOver	= pIsProcessOver;	// 进程是否结束 
	this->pIsDataTakeOff	= pIsDataTakeOff;	// 数据是否已经取走了
	this->fp16 = fp16;

	cout << "init success!" << endl;
	return true;
}

bool DataLoader::imgRead()
{
	TinyTIFFReaderFile* tiffr = NULL;
	tiffr = TinyTIFFReader_open(this->fileName.c_str());		// 打开文件
	if (!tiffr) {
		std::cout << "    ERROR reading (not existent, not accessible or no TIFF file)\n";
	}
	else
	{
		if (TinyTIFFReader_wasError(tiffr))
		{
			std::cout << "   ERROR:" << TinyTIFFReader_getLastError(tiffr) << "\n";
		}
		else
		{
			std::cout << "    ImageDescription:\n" << TinyTIFFReader_getImageDescription(tiffr) << "\n";
			imageTotal = TinyTIFFReader_countFrames(tiffr);				// 图像总数	
			std::cout << "    frames: " << imageTotal << "\n";

			const uint16_t samples = TinyTIFFReader_getSamplesPerPixel(tiffr);			// 图像每个像素的采样个数
			const uint16_t bitspersample = TinyTIFFReader_getBitsPerSample(tiffr, 0);	// 第i号通道的位数
			std::cout << "    each pixel has " << samples << " samples with " << bitspersample << " bits each\n";

			imageWidth	= TinyTIFFReader_getWidth(tiffr);				// 图像宽高
			imageHeight = TinyTIFFReader_getHeight(tiffr);

			// 创建二维数组，并往里面填写数据
			if (!fp16)
				this->imgRaw = new float[batchSize*imageHeight*imageWidth];					// 多少个batch
			else
				this->imgRaw_fp16 = new half[batchSize*imageHeight*imageWidth];				// 

			imgTemp = new unsigned short[imageHeight*imageWidth];
			// 依次读取batch size的数据
			while (curFrame < (curBatch + 1)*batchSize && curFrame < imageTotal)					// 小于batchSize大小，小于图像总数
			{
				if (curFrame % batchSize == 0)
				{
					curBatch++;
					//cout << "*************************当前batch：" << curBatch << "*********************" << endl;
				}

				// 阻塞，所有arr都被填满了
				std::unique_lock<mutex> sbguard1(*this->pMutex);
				this->pCondVal->wait(sbguard1, [this] {
					if (curFrame != 0 && curFrame % batchSize == 0 && !*pIsDataTakeOff )		// 数据已准备好，但还没拿走
					{
						*pIsArrFull = true;
						pCondVal->notify_one();				// 通知可以来拿数据了
						//cout << "threadID:"<<std::this_thread::get_id()<<"可以来拿数据了" << endl;
						return false;						// 已经填满了,阻塞
					}

					*pIsDataTakeOff = false;	
					return true;
				});

				//cout << "threadID:" << std::this_thread::get_id() << "阻塞状态解开了" << endl;
				//	TinyTIFFReader_getSampleData(tiffr, imgRaw + (curFrame%batchSize) * imageWidth*imageHeight, 0);		// 读取通道0的数据，只有一个通道
				TinyTIFFReader_getSampleData(tiffr, imgTemp, 0);		// 读取通道0的数据，只有一个通道
				imageNormalize(curFrame);								// 归一化，并将数据拷贝到imgRaw中
				sbguard1.unlock();

				//cout << "threadID:" << std::this_thread::get_id() << "当前帧数：" << curFrame << endl;

				if (TinyTIFFReader_hasNext(tiffr))				// 图像索引移到下一张图像
				{
					TinyTIFFReader_readNext(tiffr);				// 读取下一帧,指针移动到下一帧图像
					curFrame++;									// 下一帧	
				}
				else
				{
					int imgLeftNums = batchSize - imageTotal % batchSize;
					if (imgLeftNums == batchSize) imgLeftNums = 0;

					// 竞争互斥锁
					std::unique_lock<mutex> sbguard1(*this->pMutex);
					if(!fp16)
						memset(imgRaw + ((curFrame + 1) % batchSize) * imageWidth * imageHeight, 0, imgLeftNums * imageWidth * imageHeight * sizeof(float));	// 最后一个batch剩余图像全部置为0
					else
						memset(imgRaw_fp16 + ((curFrame + 1) % batchSize) * imageWidth * imageHeight, 0, imgLeftNums * imageWidth * imageHeight * sizeof(half_float::half));	// 最后一个batch剩余图像全部置为0
					this->pCondVal->wait(sbguard1, [this]{	
						if (!*pIsDataTakeOff)		// 数据已满，没有拿走
						{
							*pIsArrFull = true;
							pCondVal->notify_one();	// 通知可以来拿数据了
							return false;			// 阻塞
						}
						return true;
					});				

					//cout << "threadID:" << std::this_thread::get_id() << "剩余帧数：" << imgLeftNums << endl;
					curFrame++;									// 退出当前循环
					*pIsProcessOver = true;		// 当前进程可结束了

					//Mat img(imageHeight*batchSize, imageWidth, CV_32FC1, imgRaw);
					//imwrite("img.tif", img);
				}

				//Mat img(imageHeight*batchSize, imageWidth, CV_32FC1, imgRaw);
				//imwrite("img.tif", img);
			}
		}
	}
	return true;
}

bool DataLoader::deInit()
{
	// 释放分配的内存
	delete[] imgRaw;
	delete[] imgTemp;
	return true;
}

bool DataLoader::imageNormalize(int curFrame)
{
	// 最大最小值查找
	imgMaxMinFind();			

	// 归一化
	for (int row = 0; row < imageHeight; row++)
	{
		for (int col=0; col < imageWidth; col++)
		{
			if (!fp16)
			{
				float value = imgTemp[row*imageWidth + col];
				this->imgRaw[(curFrame%batchSize) * imageWidth*imageHeight + row * imageWidth + col] = (value - imgMinV) / (imgMaxV - imgMinV);
			}
			else  // fp16
			{
				float value		= imgTemp[row*imageWidth + col];
				half normValue  = half((value - imgMinV) / (imgMaxV - imgMinV));
				this->imgRaw_fp16[(curFrame%batchSize) * imageWidth*imageHeight + row * imageWidth + col] = normValue;
			}
		}
	}
	return true;
}

// 最大最小值查找
void DataLoader::imgMaxMinFind()
{
	for (int row = 0; row < imageHeight; row++)
	{
		for (int col=0; col < imageWidth; col++)
		{
			if (imgTemp[row*imageWidth + col] > imgMaxV)
				imgMaxV = imgTemp[row*imageWidth + col];

			if (imgTemp[row*imageWidth + col] < imgMinV)
				imgMinV = imgTemp[row*imageWidth + col];
		}
	}
	return;
}

int main_dataloader_00()
{
	// 多线程
	std::mutex* pMutex = new std::mutex;								// 互斥量
	std::condition_variable* pCondVal = new std::condition_variable;	// 条件变量
	bool isArrFull		= false;
	bool isProcessOver	= false;										// 进程是否结束
	bool isDataTakeOff	= false;

	DataLoader dataloader;
	string fileName = "D:\\project\\Pro7-denseDL\\data\\expriment\\data2\\test\\MMStack_Pos-1_metadata-256x256\\temp1.tif";
	bool fp16 = false;
	dataloader.init(fileName, 13,pMutex,pCondVal,&isArrFull,&isProcessOver,&isDataTakeOff,fp16);

	dataloader.imgRead();
	dataloader.deInit();

	system("pause");
	return 0;  // 返回真值
}