#pragma once
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include <iostream>

using namespace std;
#include <chrono>

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <malloc.h>
#include "parameters.h"
#include "DataLoader.h"

using namespace half_float;
using samplesCommon::SampleUniquePtr;

/*
�ο����ף�
��1-���л�ȥ���л�����https://blog.csdn.net/qq_35054151/article/details/111767750
*/

class SampleOnnxMNIST
{
public:
	SampleOnnxMNIST(){}	// SampleOnnxMNIST��Ĺ��캯��������һ��sampleCommon::OnnxSampleParams����Ϊ��������������ʼ����ԱmParams��Ĭ�ϳ�ԱmEngineΪnullptr

	//���� ������ʼ��
	bool initializeSampleParams(string inputDataDir, string outputDataDir, int scaleFactor, int modeType, std::mutex* pMutex, std::condition_variable* pCondVal, bool* pIsArrFull, bool* pIsProcessOver, bool* pIsDataTakeOff, bool* pIsSetimgRawToHostBuffer, bool fp16);

	//���� �������л����磬���ģ�ʹ��ڣ����������л���ȥ���л�
	bool build();

	//���� �ƶ�����
	bool infer();

	//���� ȡ�����ݲ���Ⱦ

	bool deinit();

public:
	Params mParams;				//!< The parameters for the sample.	�����̲���
	DataLoader dataloader;		// ���ݼ�����

	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.	ά����Ϣ���ࣩ
	nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
	int mNumber{ 0 };             //!< The number to classify							ͼ�����ʵ����

	std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr; //!< The TensorRT engine used to run the network	����ָ�룺TensorRT����


	//samplesCommon::ManagedBuffer mInput{};          //!< Host and device buffers for the input.
	//samplesCommon::ManagedBuffer mOutput{};         //!< Host and device buffers for the ouptut.
	std::unique_ptr<samplesCommon::ManagedBuffer> mInput{ new samplesCommon::ManagedBuffer() };		// ����buff ָ��
	std::unique_ptr<samplesCommon::ManagedBuffer> mOutput{ new samplesCommon::ManagedBuffer() };		// ���buff ָ��

	//!
	//! \brief Parses an ONNX model for MNIST and creates a TensorRT network
	//!
	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
		SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
		SampleUniquePtr<nvonnxparser::IParser>& parser);

	//! \brief Reads the input  and stores the result in a managed buffer
	bool processInput(const samplesCommon::BufferManager& buffers);

	//! \brief Classifies digits and verify result
	bool verifyOutput(const samplesCommon::BufferManager& buffers);

	// �������� zzw
	bool verifyOutput(const samplesCommon::ManagedBuffer& buffers, TinyTIFFWriterFile* tif);

	// �ļ��Ƿ����
	bool isFileExists_ifstream(string& name)
	{
		ifstream f(name.c_str());
		return f.good();
	}

	bool normImgSave();
	bool normImgSaveOutput();

	// ���߳�
	std::mutex* pMutex;					//������������
	std::condition_variable* pCondVal;
	bool* pIsArrFull		= nullptr;
	bool* pIsSetimgRawToHostBuffer = nullptr;	// �Ƿ��������imgRawToHostBuffer��ָ�뿽����

	bool* pIsProcessOver	= nullptr;
	bool* pIsDataTakeOff	= nullptr; // �����Ƿ��Ѿ�ȡ����

	half* hostDataBuffer_half	= nullptr;
	float* hostDataBuffer		= nullptr;
	

};
