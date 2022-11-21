#include<iostream>
#include<thread>
#include<mutex>
#include"sampleOnnxMNIST.h"
#include"DataLoader.h"

using namespace std;

int main(int argc, char** argv)
{
	// ���߳�
	std::mutex* pMutex = new std::mutex;								// ������
	std::condition_variable* pCondVal = new std::condition_variable;	// ��������
	bool isArrFull = false;	// batch �����Ƿ��Ѿ����ˣ�����ȡ����
	bool isDataTakeOff = false;	// �����Ƿ��Ѿ�ȡ����
	bool isProcessOver = false;	// ���̽���

	/**************************���ݼ��س�ʼ��**********************************/
	SampleOnnxMNIST sample;
	string inputDataDir = "C:\\Users\\JN Wu\\Desktop\\cpp\\cpp\\data";
	string outputDataDir= "C:\\Users\\JN Wu\\Desktop\\cpp\\cpp\\result";

	string fileName = inputDataDir + "\\" + "rawImg_256x256.tif";
	int batchSize	= 13;
	bool fp16		= false;
	int modelType	= 256;
	int scaleFactor = 8;

	sample.dataloader.init(fileName, batchSize, pMutex, pCondVal, &isArrFull, &isProcessOver, &isDataTakeOff, fp16);
	/**************************�����ƶϳ�ʼ��**********************************/
	auto sampleTest = sample::gLogger.defineTest("my tensorRT", argc, argv);	// ����һ����־��
	sample::gLogger.reportTestStart(sampleTest);								// ��¼��־�Ŀ�ʼ

	// ������������
	sample.initializeSampleParams(inputDataDir, outputDataDir, scaleFactor, modelType, pMutex, pCondVal, &isArrFull, &isProcessOver, &isDataTakeOff, fp16);	// ��ʼ������		

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

	cout << "total time��" << totalTime << " ms" << endl;
	/**************************�ڴ��ͷ�**********************************/
	sample.dataloader.deInit();
	delete pMutex;				// �ͷŻ�����
	delete pCondVal;			// �ͷ���������

	sample.deinit();

	//system("pause");
	return 0;
}