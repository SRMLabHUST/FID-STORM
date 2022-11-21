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

/****************************DataLoader�ĺ���ʵ��*********************************************/
bool DataLoader::init(string fileName, int batchSize, std::mutex* pMutex, std::condition_variable* pCondVal,bool* pIsArrFull,bool * pIsProcessOver, bool* pIsDataTakeOff,bool fp16)
{
	this->fileName  = fileName;
	this->batchSize = batchSize;

	this->pMutex			= pMutex;			// �������������������죩
	this->pCondVal			= pCondVal;			// ��������
	this->pIsArrFull		= pIsArrFull;		// �����Ƿ�����
	this->pIsProcessOver	= pIsProcessOver;	// �����Ƿ���� 
	this->pIsDataTakeOff	= pIsDataTakeOff;	// �����Ƿ��Ѿ�ȡ����
	this->fp16 = fp16;

	cout << "init success!" << endl;
	return true;
}

bool DataLoader::imgRead()
{
	TinyTIFFReaderFile* tiffr = NULL;
	tiffr = TinyTIFFReader_open(this->fileName.c_str());		// ���ļ�
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
			imageTotal = TinyTIFFReader_countFrames(tiffr);				// ͼ������	
			std::cout << "    frames: " << imageTotal << "\n";

			const uint16_t samples = TinyTIFFReader_getSamplesPerPixel(tiffr);			// ͼ��ÿ�����صĲ�������
			const uint16_t bitspersample = TinyTIFFReader_getBitsPerSample(tiffr, 0);	// ��i��ͨ����λ��
			std::cout << "    each pixel has " << samples << " samples with " << bitspersample << " bits each\n";

			imageWidth	= TinyTIFFReader_getWidth(tiffr);				// ͼ����
			imageHeight = TinyTIFFReader_getHeight(tiffr);

			// ������ά���飬����������д����
			if (!fp16)
				this->imgRaw = new float[batchSize*imageHeight*imageWidth];					// ���ٸ�batch
			else
				this->imgRaw_fp16 = new half[batchSize*imageHeight*imageWidth];				// 

			imgTemp = new unsigned short[imageHeight*imageWidth];
			// ���ζ�ȡbatch size������
			while (curFrame < (curBatch + 1)*batchSize && curFrame < imageTotal)					// С��batchSize��С��С��ͼ������
			{
				if (curFrame % batchSize == 0)
				{
					curBatch++;
					//cout << "*************************��ǰbatch��" << curBatch << "*********************" << endl;
				}

				// ����������arr����������
				std::unique_lock<mutex> sbguard1(*this->pMutex);
				this->pCondVal->wait(sbguard1, [this] {
					if (curFrame != 0 && curFrame % batchSize == 0 && !*pIsDataTakeOff )		// ������׼���ã�����û����
					{
						*pIsArrFull = true;
						pCondVal->notify_one();				// ֪ͨ��������������
						//cout << "threadID:"<<std::this_thread::get_id()<<"��������������" << endl;
						return false;						// �Ѿ�������,����
					}

					*pIsDataTakeOff = false;	
					return true;
				});

				//cout << "threadID:" << std::this_thread::get_id() << "����״̬�⿪��" << endl;
				//	TinyTIFFReader_getSampleData(tiffr, imgRaw + (curFrame%batchSize) * imageWidth*imageHeight, 0);		// ��ȡͨ��0�����ݣ�ֻ��һ��ͨ��
				TinyTIFFReader_getSampleData(tiffr, imgTemp, 0);		// ��ȡͨ��0�����ݣ�ֻ��һ��ͨ��
				imageNormalize(curFrame);								// ��һ�����������ݿ�����imgRaw��
				sbguard1.unlock();

				//cout << "threadID:" << std::this_thread::get_id() << "��ǰ֡����" << curFrame << endl;

				if (TinyTIFFReader_hasNext(tiffr))				// ͼ�������Ƶ���һ��ͼ��
				{
					TinyTIFFReader_readNext(tiffr);				// ��ȡ��һ֡,ָ���ƶ�����һ֡ͼ��
					curFrame++;									// ��һ֡	
				}
				else
				{
					int imgLeftNums = batchSize - imageTotal % batchSize;
					if (imgLeftNums == batchSize) imgLeftNums = 0;

					// ����������
					std::unique_lock<mutex> sbguard1(*this->pMutex);
					if(!fp16)
						memset(imgRaw + ((curFrame + 1) % batchSize) * imageWidth * imageHeight, 0, imgLeftNums * imageWidth * imageHeight * sizeof(float));	// ���һ��batchʣ��ͼ��ȫ����Ϊ0
					else
						memset(imgRaw_fp16 + ((curFrame + 1) % batchSize) * imageWidth * imageHeight, 0, imgLeftNums * imageWidth * imageHeight * sizeof(half_float::half));	// ���һ��batchʣ��ͼ��ȫ����Ϊ0
					this->pCondVal->wait(sbguard1, [this]{	
						if (!*pIsDataTakeOff)		// ����������û������
						{
							*pIsArrFull = true;
							pCondVal->notify_one();	// ֪ͨ��������������
							return false;			// ����
						}
						return true;
					});				

					//cout << "threadID:" << std::this_thread::get_id() << "ʣ��֡����" << imgLeftNums << endl;
					curFrame++;									// �˳���ǰѭ��
					*pIsProcessOver = true;		// ��ǰ���̿ɽ�����

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
	// �ͷŷ�����ڴ�
	delete[] imgRaw;
	delete[] imgTemp;
	return true;
}

bool DataLoader::imageNormalize(int curFrame)
{
	// �����Сֵ����
	imgMaxMinFind();			

	// ��һ��
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

// �����Сֵ����
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
	// ���߳�
	std::mutex* pMutex = new std::mutex;								// ������
	std::condition_variable* pCondVal = new std::condition_variable;	// ��������
	bool isArrFull		= false;
	bool isProcessOver	= false;										// �����Ƿ����
	bool isDataTakeOff	= false;

	DataLoader dataloader;
	string fileName = "D:\\project\\Pro7-denseDL\\data\\expriment\\data2\\test\\MMStack_Pos-1_metadata-256x256\\temp1.tif";
	bool fp16 = false;
	dataloader.init(fileName, 13,pMutex,pCondVal,&isArrFull,&isProcessOver,&isDataTakeOff,fp16);

	dataloader.imgRead();
	dataloader.deInit();

	system("pause");
	return 0;  // ������ֵ
}