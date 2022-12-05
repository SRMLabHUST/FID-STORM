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
参考文献：
【1-序列化去序列化】：https://blog.csdn.net/qq_35054151/article/details/111767750
*/

class SampleOnnxMNIST
{
public:
	SampleOnnxMNIST(){}	// SampleOnnxMNIST类的构造函数，接收一个sampleCommon::OnnxSampleParams类作为参数，并用来初始化成员mParams，默认成员mEngine为nullptr

	//【】 参数初始化
	bool initializeSampleParams(string inputDataDir, string outputDataDir, int scaleFactor,int modeType,std::mutex* pMutex, std::condition_variable* pCondVal, bool* pIsArrFull,bool* pIsProcessOver,bool* pIsDataTakeOff,bool fp16);

	//【】 构建序列化网络，如果模型存在，则跳过序列化，去序列化
	bool build();

	//【】 推断网络
	bool infer();

	//【】 取出数据并渲染

	bool deinit();

public:
	Params mParams;				//!< The parameters for the sample.	本例程参数
	DataLoader dataloader;		// 数据加载类

	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.	维度信息（类）
	nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
	int mNumber{ 0 };             //!< The number to classify							图像的真实数字

	std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr; //!< The TensorRT engine used to run the network	智能指针：TensorRT引擎


	//samplesCommon::ManagedBuffer mInput{};          //!< Host and device buffers for the input.
	//samplesCommon::ManagedBuffer mOutput{};         //!< Host and device buffers for the ouptut.
	std::unique_ptr<samplesCommon::ManagedBuffer> mInput{ new samplesCommon::ManagedBuffer() };		// 输入buff 指针
	std::unique_ptr<samplesCommon::ManagedBuffer> mOutput{ new samplesCommon::ManagedBuffer() };		// 输出buff 指针

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

	// 函数重载 zzw
	bool verifyOutput(const samplesCommon::ManagedBuffer& buffers, TinyTIFFWriterFile* tif);

	// 文件是否存在
	bool isFileExists_ifstream(string& name)
	{
		ifstream f(name.c_str());
		return f.good();
	}

	bool normImgSave();
	bool normImgSaveOutput();

	// 多线程
	std::mutex* pMutex;					//不允许拷贝构造
	std::condition_variable* pCondVal;
	bool* pIsArrFull		= nullptr;
	bool* pIsProcessOver	= nullptr;
	bool* pIsDataTakeOff	= nullptr; // 数据是否已经取走了

	half* hostDataBuffer_half	= nullptr;
	float* hostDataBuffer		= nullptr;
	

};
