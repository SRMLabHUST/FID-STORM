#include<iostream>
#include<thread>
#include<mutex>
#include"sampleOnnxMNIST.h"
#include"DataLoader.h"

using namespace std;

int main00(int argc, char** argv,string inputDataDir, string outputDataDir, int batchSize, int modelType, int scaleFactor, bool fp16)
{
	// ���߳�
	std::mutex* pMutex = new std::mutex;								// Mutex quantity
	std::condition_variable* pCondVal = new std::condition_variable;	// Conditional variable
	bool isArrFull			= false;	// If the batch array is full. If full, it can fetch data
	bool isDataTakeOff		= false;	// Whether the data has been taken
	bool isProcessOver		= false;	// Process is over

	/**************************Initialization of data load**********************************/
	SampleOnnxMNIST sample;

	string fileName = inputDataDir + "\\"+ "rawImg_256x256.tif";

	sample.dataloader.init(fileName, batchSize,pMutex,pCondVal, &isArrFull,&isProcessOver,&isDataTakeOff,fp16);
	/**************************Initialization of network inference**********************************/
	auto sampleTest = sample::gLogger.defineTest("my tensorRT", argc, argv);	// ����һ����־��
	sample::gLogger.reportTestStart(sampleTest);								// ��¼��־�Ŀ�ʼ

	// ������������
	sample.initializeSampleParams(inputDataDir, outputDataDir,scaleFactor,modelType,pMutex, pCondVal,&isArrFull,&isProcessOver, &isDataTakeOff,fp16);	// ��ʼ������		

	// ������������
	if (!sample.build())	return sample::gLogger.reportFail(sampleTest);		// 

	/**************************�����ƶ�ִ��**********************************/
	auto startTime = chrono::high_resolution_clock::now();	// ��ǰʱ��
	// �����ƶϣ����߳�
	thread myThreadDataLoad(&DataLoader::imgRead, std::ref(sample.dataloader));	// �߳�1�������������м�������
	thread myThreadInfer(&SampleOnnxMNIST::infer, std::ref(sample));			// �߳�2������ȡ�����ݣ���������
	
	myThreadDataLoad.join();
	myThreadInfer.join();

	auto  endTime = chrono::high_resolution_clock::now();
	float totalTime = chrono::duration<float, milli>(endTime - startTime).count();

	cout << "total time��" << totalTime  <<" ms"<< endl;
	/**************************�ڴ��ͷ�**********************************/
	sample.dataloader.deInit();
	delete pMutex;				// �ͷŻ�����
	delete pCondVal;			// �ͷ���������

	sample.deinit();

	//system("pause");
	return 0;
}