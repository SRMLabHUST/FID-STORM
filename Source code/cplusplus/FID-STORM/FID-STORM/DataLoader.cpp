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

/****************************The function implementation of DataLoader*********************************************/
bool DataLoader::init(string fileName, int batchSize, std::mutex* pMutex, std::condition_variable* pCondVal,bool* pIsArrFull,bool * pIsProcessOver, bool* pIsDataTakeOff,bool fp16)
{
	this->fileName  = fileName;
	this->batchSize = batchSize;

	this->pMutex			= pMutex;			// Mutex (copy constructs are not allowed)
	this->pCondVal			= pCondVal;			// Conditional variable
	this->pIsArrFull		= pIsArrFull;		// Whether the array is full
	this->pIsProcessOver	= pIsProcessOver;	// Whether the process is over 
	this->pIsDataTakeOff	= pIsDataTakeOff;	// Whether the data has been taken
	this->fp16 = fp16;

	cout << "init success!" << endl;
	return true;
}

bool DataLoader::imgRead()
{
	TinyTIFFReaderFile* tiffr = NULL;
	tiffr = TinyTIFFReader_open(this->fileName.c_str());		// Open file
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
			imageTotal = TinyTIFFReader_countFrames(tiffr);				// The total of images	
			std::cout << "    frames: " << imageTotal << "\n";

			const uint16_t samples = TinyTIFFReader_getSamplesPerPixel(tiffr);			// The number of samples per pixel of the image
			const uint16_t bitspersample = TinyTIFFReader_getBitsPerSample(tiffr, 0);	// The number of bits in channel i
			std::cout << "    each pixel has " << samples << " samples with " << bitspersample << " bits each\n";

			imageWidth	= TinyTIFFReader_getWidth(tiffr);				// Image width and height
			imageHeight = TinyTIFFReader_getHeight(tiffr);

			// Create a two-dimensional array and fill it with data
			if (!fp16)
				this->imgRaw = new float[batchSize*imageHeight*imageWidth];					// The number of batch
			else
				this->imgRaw_fp16 = new half[batchSize*imageHeight*imageWidth];				

			imgTemp = new unsigned short[imageHeight*imageWidth];
			// Read batch size data in sequence
			while (curFrame < (curBatch + 1)*batchSize && curFrame < imageTotal)			// Smaller than batchSize, and smaller than the total number of images
			{
				if (curFrame % batchSize == 0)
				{
					curBatch++;
					//cout << "*************************Current batch£º" << curBatch << "*********************" << endl;
				}

				// Blocked. All arr are filled
				std::unique_lock<mutex> sbguard1(*this->pMutex);
				this->pCondVal->wait(sbguard1, [this] {
					if (curFrame != 0 && curFrame % batchSize == 0 && !*pIsDataTakeOff )		// The data is ready, but hasn't been taken
					{
						*pIsArrFull = true;
						pCondVal->notify_one();				// Data is availavle and ready to pick up 
						//cout << "threadID:"<<std::this_thread::get_id()<<"Data is availavle and ready to pick up " << endl;
						return false;						// It's full, blocked
					}

					*pIsDataTakeOff = false;	
					return true;
				});

				//cout << "threadID:" << std::this_thread::get_id() << "The blockage is unblocked" << endl;
				//	TinyTIFFReader_getSampleData(tiffr, imgRaw + (curFrame%batchSize) * imageWidth*imageHeight, 0);		// Read the data of channel 0. There is only one channel
				TinyTIFFReader_getSampleData(tiffr, imgTemp, 0);		// Read the data of channel 0. There is only one channel
				imageNormalize(curFrame);								// Normalize ,and copy the data to imgRaw
				sbguard1.unlock();

				//cout << "threadID:" << std::this_thread::get_id() << "Current frame£º" << curFrame << endl;

				if (TinyTIFFReader_hasNext(tiffr))				// The image index moves to the next image
				{
					TinyTIFFReader_readNext(tiffr);				// Read the next frame, the pointer moves to the next frame image
					curFrame++;									// The next frame	
				}
				else
				{
					int imgLeftNums = batchSize - imageTotal % batchSize;
					if (imgLeftNums == batchSize) imgLeftNums = 0;

					// Competing mutex
					std::unique_lock<mutex> sbguard1(*this->pMutex);
					if(!fp16)
						memset(imgRaw + ((curFrame + 1) % batchSize) * imageWidth * imageHeight, 0, imgLeftNums * imageWidth * imageHeight * sizeof(float));	// All the remaining images of the last batch are set to 0
					else
						memset(imgRaw_fp16 + ((curFrame + 1) % batchSize) * imageWidth * imageHeight, 0, imgLeftNums * imageWidth * imageHeight * sizeof(half_float::half));	// All the remaining images of the last batch are set to 0
					this->pCondVal->wait(sbguard1, [this]{	
						if (!*pIsDataTakeOff)		// The data is full and not taken away
						{
							*pIsArrFull = true;
							pCondVal->notify_one();	// The data is ready to pick up
							return false;			// Blocking
						}
						return true;
					});				

					//cout << "threadID:" << std::this_thread::get_id() << "Remaining frame£º" << imgLeftNums << endl;
					curFrame++;									// Exit the current loop
					*pIsProcessOver = true;		// The current process is finished

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
	// Frees allocated memory
	delete[] imgRaw;
	delete[] imgTemp;
	return true;
}

bool DataLoader::imageNormalize(int curFrame)
{
	// Max-min search
	imgMaxMinFind();			

	// Normalization
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

// Max-min search
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
	// Multiple thread
	std::mutex* pMutex = new std::mutex;								// Mutex quantity
	std::condition_variable* pCondVal = new std::condition_variable;	// Conditional variable
	bool isArrFull		= false;
	bool isProcessOver	= false;										// Whether the process is over
	bool isDataTakeOff	= false;

	DataLoader dataloader;
	string fileName = "D:\\project\\Pro7-denseDL\\data\\expriment\\data2\\test\\MMStack_Pos-1_metadata-256x256\\temp1.tif";
	bool fp16 = false;
	dataloader.init(fileName, 13,pMutex,pCondVal,&isArrFull,&isProcessOver,&isDataTakeOff,fp16);

	dataloader.imgRead();
	dataloader.deInit();

	system("pause");
	return 0;  // Return true value
}