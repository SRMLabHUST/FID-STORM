#include<iostream>
#include<thread>
#include<mutex>
#include"sampleOnnxMNIST.h"
#include"DataLoader.h"

using namespace std;

int main00(int argc, char** argv,string inputDataDir, string outputDataDir, int batchSize, int modelType, int scaleFactor, bool fp16)
{
	// 多线程
	std::mutex* pMutex = new std::mutex;								// 互斥量
	std::condition_variable* pCondVal = new std::condition_variable;	// 条件变量
	bool isArrFull			= false;	// batch 数组是否已经满了，可以取数据
	bool isDataTakeOff		= false;	// 数据是否已经取走了
	bool isProcessOver		= false;	// 进程结束

	/**************************数据加载初始化**********************************/
	SampleOnnxMNIST sample;
	//string inputDataDir = "C:\\Users\\JN Wu\\Desktop\\cpp\\cpp\\data";
	//string outputDataDir= "C:\\Users\\JN Wu\\Desktop\\cpp\\cpp\\result";

	string fileName = inputDataDir + "\\"+ "rawImg_256x256.tif";
	//int batchSize	= 13;
	//bool fp16		= false;
	//int modelType	= 256;
	//int scaleFactor = 8;

	sample.dataloader.init(fileName, batchSize,pMutex,pCondVal, &isArrFull,&isProcessOver,&isDataTakeOff,fp16);
	/**************************网络推断初始化**********************************/
	auto sampleTest = sample::gLogger.defineTest("my tensorRT", argc, argv);	// 定义一个日志类
	sample::gLogger.reportTestStart(sampleTest);								// 记录日志的开始

	// 【】参数解析
	sample.initializeSampleParams(inputDataDir, outputDataDir,scaleFactor,modelType,pMutex, pCondVal,&isArrFull,&isProcessOver, &isDataTakeOff,fp16);	// 初始化参数		

	// 【】构建网络
	if (!sample.build())	return sample::gLogger.reportFail(sampleTest);		// 

	/**************************网络推断执行**********************************/
	auto startTime = chrono::high_resolution_clock::now();	// 当前时钟
	// 【】推断，多线程
	thread myThreadDataLoad(&DataLoader::imgRead, std::ref(sample.dataloader));	// 线程1，不断往数组中加载数据
	thread myThreadInfer(&SampleOnnxMNIST::infer, std::ref(sample));			// 线程2，不断取出数据，并作推理
	
	myThreadDataLoad.join();
	myThreadInfer.join();

	auto  endTime = chrono::high_resolution_clock::now();
	float totalTime = chrono::duration<float, milli>(endTime - startTime).count();

	cout << "total time：" << totalTime  <<" ms"<< endl;
	/**************************内存释放**********************************/
	sample.dataloader.deInit();
	delete pMutex;				// 释放互斥量
	delete pCondVal;			// 释放条件变量

	sample.deinit();

	//system("pause");
	return 0;
}