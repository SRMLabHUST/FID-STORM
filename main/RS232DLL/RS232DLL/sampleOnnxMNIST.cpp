
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include <iostream>

#include <chrono>

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <malloc.h>
#include "parameters.h"
#include "sampleOnnxMNIST.h"
#include "tinytiffwriter.h"
#include "tinytiffreader.h"
#include "cudaImageAdd.h"
#include"device_launch_parameters.h"

using namespace std;
using namespace half_float;
using samplesCommon::SampleUniquePtr;

bool SampleOnnxMNIST::build()
{
	// 【1】创建Builder(用来优化模型+产生Engine)
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));  
    if (!builder)
    {
        return false;
    }

	if (builder->platformHasFastFp16())
		cout << "当前平台支持fp16" << endl;
	else
		cout << "当前平台不支持fp16" << endl;

	// 【1.1】使用builder创建Network
		// 1) 定义网络
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

	// 【1.2】使用builder创建config
		// 1) 选择运算精度(FP16 or INT8)
		// 2) 优化模型 （消除无效计算、折叠张量、重新排序）
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

	// 使用 FP16
	if (mParams.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}

	// 【3】创建优化配置文件，并更新配置
	auto profile = builder->createOptimizationProfile();
	const auto inputName = mParams.inputTensorNames[0].c_str();
	//mInputDims.d[0] = 2; mInputDims.d[1]  = 1;	mInputDims.d[2]  = 128;	 mInputDims.d[3]  = 128;
	//mOutputDims.d[0]= 2; mOutputDims.d[1] = 2;	mOutputDims.d[2] = 1024; mOutputDims.d[3] = 1024;

	mInputDims	= Dims4(mParams.batchSize, 1, mParams.inputWidth, mParams.inputHight);
	mOutputDims = Dims4(mParams.batchSize, 1, mParams.outputWidth,mParams.outputHight);

	// This profile will be valid for all images whose size falls in the range of [(1, 1, 1, 1), (1, 1, 56, 56)]
	// but TensorRT will optimize for (1, 1, 28, 28)
	// We do not need to check the return of setDimension and addOptimizationProfile here as all dims are explicitly set
	profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(mParams.batchSize, 1, mParams.inputWidth, mParams.inputHight));
	profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(mParams.batchSize, 1, mParams.inputWidth, mParams.inputHight));
	profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(mParams.batchSize, 1, mParams.inputWidth, mParams.inputHight));
	config->addOptimizationProfile(profile);

	//// 新的配置指针
	//auto profileCalib = builder->createOptimizationProfile();
	//// We do not need to check the return of setDimension and setCalibrationProfile here as all dims are explicitly set
	//profileCalib->setDimensions(inputName, OptProfileSelector::kMIN, mInputDims);
	//profileCalib->setDimensions(inputName, OptProfileSelector::kOPT, mInputDims);
	//profileCalib->setDimensions(inputName, OptProfileSelector::kMAX, mInputDims);
	//config->setCalibrationProfile(profileCalib);

	// 【2】定义parser，此处用来解析onnx文件
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

	// 【3】构建引擎 engine (constructed)
    auto constructed = constructNetwork(builder, network, config, parser);  // 进入函数
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
	// builder，用于分析cuda流
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

	// 构建序列化网络
	if (!isFileExists_ifstream(mParams.engine_file))		// trt文件不存在
	{
		ofstream outfile(mParams.engine_file.c_str(), ios::out | ios::binary);		// 创建文件
		if (!outfile.is_open())	// 文件打开失败
		{
			fprintf(stderr, "fail to open file to write:%s\n", mParams.engine_file.c_str());
			return false;
		}

		// 编译引擎
		SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
		if (!plan)
		{
			return false;
		}

		// 保存文件
		fprintf(stdout, "allocate memory size:%d byte\n", plan->size());							// 构建引擎大小
		unsigned char * p = (unsigned char*)plan->data();
		outfile.write((char*)p, plan->size());
		outfile.close();
	}

	// 创建推断运行环境
    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

	// 去序列化cuda引擎
		// 注意：去序列化，第一次推断数据，模型需要加载一下，有额外的处理时间，因此，第一次测量的时间会较长；多测几次，可以看到后面推断时，时间正常
	ifstream fin(mParams.engine_file,ios::in|ios::binary);
	string cached_engine = "";
	while (fin.peek() != EOF)
	{
		stringstream buffer;
		buffer << fin.rdbuf();						// 把infile流对象中的流重定向到标准输出cout上
		cached_engine.append(buffer.str());
	}

	fin.close();

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

	// 需要根据网络的输入输出Tensor的形状进行调整
    //ASSERT(network->getNbInputs() == 1);					// 保证输入batchsize为1
    //mInputDims = network->getInput(0)->getDimensions();		// 获取输入维度
    //ASSERT(mInputDims.nbDims == 4);							// 保证输入维度为4

    //ASSERT(network->getNbOutputs() == 1);					// 保证输出batchsize为1
    //mOutputDims = network->getOutput(0)->getDimensions();	// 获取输出维度
    //ASSERT(mOutputDims.nbDims == 2);						// 保证输出维度为2

    return true;
}

bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
	//解析 onnxfile
    //auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
    //    static_cast<int>(sample::gLogger.getReportableSeverity()));
	auto parsed = parser->parseFromFile(mParams.onnxFileName.c_str(),static_cast<int>(sample::gLogger.getReportableSeverity()));

    if (!parsed)
    {
        return false;
    }

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

bool SampleOnnxMNIST::infer()
{
	// 【】创建执行context
    // Create RAII buffer manager object 
		// 1) 创建缓存
		// 2）创建执行的 “context”
    samplesCommon::BufferManager buffers(mEngine);								// 区别 BufferManager 和 MangerBuffer
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context)	return false;

	ASSERT(mParams.inputTensorNames.size() == 1);

    // 【】输入处理
		// 1）读取输入数据填充到CPU输入缓存
		// 2）执行copyInputToDevice()，自动填充到GPU输入缓存接口
	if (!processInput(buffers))	return false;

	// 【】分配内存
	mInput->deviceBuffer.resize(mInputDims);		// 输入设备内存

	// 【】输出内存分配
	//mOutput->hostBuffer.resize(mOutputDims);
	mOutput->deviceBuffer.resize(mOutputDims);		// 输出设备内存

	// 【】绑定
	// Set the input size for the preprocessor
	// 指定context输入的维度，mPreprocessorContext绑定
	CHECK_RETURN_W_MSG(context->setBindingDimensions(0, mInputDims), false, "Invalid binding dimensions.");
	// We can only run inference once all dynamic input shapes have been specified.
	// 检查，是否所有的动态输入都被指定完成
	if (!context->allInputDimensionsSpecified()) return false;
	std::vector<void*> inferBindings = { mInput->deviceBuffer.data(), mOutput->deviceBuffer.data() };

	int curBatch = 1;
	//TinyTIFFWriterFile* tif = TinyTIFFWriter_open(mParams.outputTiffName.data(), 32, TinyTIFFWriter_Float, 0, mParams.outputWidth, mParams.outputHight, TinyTIFFWriter_Greyscale);
	string outputFileName = mParams.outputDataDir + "\\renderImg.tif";
	TinyTIFFWriterFile* tif = TinyTIFFWriter_open(outputFileName.c_str(), 32, TinyTIFFWriter_Float, 0, mParams.outputWidth, mParams.outputHight, TinyTIFFWriter_Greyscale);

	// 【】主机内存 -》 设备内存
	//while (!*pIsProcessOver)

	while (!*pIsProcessOver)
	{
		std::unique_lock<mutex> sbguard1(*this->pMutex);
		this->pCondVal->wait(sbguard1, [this] {
			if (*pIsArrFull == false)						// 未满
				return false;								// 阻塞
			return true;
		});
		//cout << "threadID:" << std::this_thread::get_id() << "拿到数据了" << endl;

		auto startTime = chrono::high_resolution_clock::now();	// 当前时钟
		//Mat img(this->mParams.inputHight*this->mParams.batchSize, this->mParams.inputWidth, CV_32FC1, hostDataBuffer);
		//imwrite("img.tif", img);

		// 验证在没有溢出情况（输入全为0情况下），输出是否会出错
		//memset(hostDataBuffer_half, 0, mInput->deviceBuffer.nbBytes());

		if(!mParams.fp16)
			CHECK(cudaMemcpy(mInput->deviceBuffer.data()/*void* dst*/, hostDataBuffer/*mInput->hostBuffer.data() void* src*/, mInput->deviceBuffer.nbBytes()/*size_t count*/, cudaMemcpyHostToDevice/*cudaMemcpyKind kind*/));
		else
			CHECK(cudaMemcpy(mInput->deviceBuffer.data()/*void* dst*/, hostDataBuffer_half/*mInput->hostBuffer.data() void* src*/, mInput->deviceBuffer.nbBytes()/*size_t count*/, cudaMemcpyHostToDevice/*cudaMemcpyKind kind*/));

		//cout << _msize(hostDataBuffer_half) << endl;
		//cout << "threadID:" << std::this_thread::get_id() << "数据拷贝完了" << endl;
		
		*pIsArrFull		= false;			// 矩阵为空了
		*pIsDataTakeOff = true;				// 数据拿走了
		this->pCondVal->notify_one();		// 通知已经取得了数据
		//cout << "threadID:" << std::this_thread::get_id() << "当前帧："<< this->dataloader.curFrame << endl;
		//cout << "threadID:" << std::this_thread::get_id() << "通知解开阻塞状态了" << endl;

		//【】归一化图像的保存，测试用
		//normImgSave();
		sbguard1.unlock();
		
		//cout << "输出host的buffer大小："   << mOutput->hostBuffer.nbBytes() << endl;
		//cout << "输出device的buffer大小：" << mOutput->deviceBuffer.nbBytes() << endl;

		//normImgSaveOutput();

		// 执行
		bool status = context->executeV2(inferBindings.data());
		if (!status)
		{
			system("pause");
			return false;
		}

		// 【】拿出结果
		//imgAdd <<<blockNums, threadPerBlock>>> (mOutput->deviceBuffer.data(), mParams.renderImgOut_device, mParams.outputWidth , mParams.outputHight, mParams.batchSize);
		imgAddTop((float*)mOutput->deviceBuffer.data(), mParams.renderImgOut_device, mParams.outputWidth , mParams.outputHight, mParams.batchSize,mParams.fp16);
		//CHECK(cudaMemcpy(
		//	mOutput->hostBuffer.data()/*void* dst*/, mOutput->deviceBuffer.data()/*void* src*/, mOutput->hostBuffer.nbBytes()/*size_t count*/, cudaMemcpyDeviceToHost/*cudaMemcpyKind kind*/));

		auto  endTime = chrono::high_resolution_clock::now();
		float totalTime = chrono::duration<float, milli>(endTime - startTime).count();

		cout << "Execution time:" << totalTime << "ms, current batch:" << curBatch++ << endl;
		//cout<< "mOutput大小："<<_msize(mOutput->hostBuffer.data())<<endl;  // 正确
		//cout << "从设备到主机内存拷贝完成"  << endl;
		// 输出推断图像
		//half * p = (half*)mOutput->hostBuffer.data();
		//for (int i = 0; i < 1024*10; i++)
		//	cout << p[i] << endl;

		// 【】图像保存
		/*if (!verifyOutput(*mOutput,tif))	return false;*/
		//cout << "输出图像保存完成" << endl;
	}
	
	cudaError_t state = cudaMemcpy(mParams.renderImgOut_host, mParams.renderImgOut_device, mParams.outputWidth * mParams.outputHight * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Is output image error?:" << state << endl;
	TinyTIFFWriter_writeImage(tif, mParams.renderImgOut_host);
	TinyTIFFWriter_close(tif);			// 关闭当前图像

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[2];		// 128
    const int inputW = mInputDims.d[3];		// 128

	// 【】分配host buffer
	if (mParams.fp16)
	{
		// 输入
		mInput->hostBuffer		= samplesCommon::HostBuffer(nvinfer1::DataType::kHALF);
		mInput->deviceBuffer	= samplesCommon::DeviceBuffer(nvinfer1::DataType::kHALF);
		// 输出
		mOutput->hostBuffer		= samplesCommon::HostBuffer(nvinfer1::DataType::kHALF);
		mOutput->deviceBuffer	= samplesCommon::DeviceBuffer(nvinfer1::DataType::kHALF);
	}
	
	if (mParams.fp16)
	{
		hostDataBuffer_half = this->dataloader.imgRaw_fp16;
		//cout << "输入图像分配的主机内存：" << _msize(hostDataBuffer_half) << endl;
	}
	else
		hostDataBuffer = this->dataloader.imgRaw;

    return true;
}

bool SampleOnnxMNIST::normImgSaveOutput()
{
	//const int OutputH = mOutputDims.d[2];		// 128
	//const int OutputW = mOutputDims.d[3];		// 128

	//// 验证一：是否是输入内存的问题
	//// 证明deviceBuffer.data()中的数据是正常的
	//half* outHostDataBuffer_half = (half*)mOutput->hostBuffer.data();
	//memset(outHostDataBuffer_half, 0, mOutput->hostBuffer.nbBytes());
	//CHECK(cudaMemcpy(outHostDataBuffer_half, mOutput->deviceBuffer.data(), mOutput->deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));	
	//
	//// 分配临时内存，用于保存float32图像
	//float * imgOutTemp = new float[OutputH*OutputW];
	//for (int batch = 0; batch < mOutputDims.d[0]; batch++)
	//{
	//	// fp16 保存到fp32中，然后存图
	//	if (mParams.fp16)
	//	{
	//		for (int i = 0; i < OutputW*OutputH; i++)
	//			imgOutTemp[i] = outHostDataBuffer_half[batch*OutputW*OutputH + i];

	//		Mat imgNorm(OutputH, OutputW, CV_32FC1, imgOutTemp);
	//		if (!imwrite(mParams.outputFileName[batch], imgNorm))
	//		{
	//			cout << "imgNorm " << batch << ".tif保存失败" << endl;
	//		}
	//	}
	//	else
	//	{
	//		// 将数据保存为float32 然后保存
	//		Mat imgNorm(OutputH, OutputW, CV_32FC1, outHostDataBuffer_half + batch * OutputH*OutputW);
	//		if (!imwrite(mParams.outputFileName[batch], imgNorm))
	//		{
	//			cout << "imgNorm " << batch << ".tif保存失败" << endl;
	//		}
	//	}
	//}
	//delete imgOutTemp;
	return true;
}


bool SampleOnnxMNIST::normImgSave()
{
	//const int inputH = mInputDims.d[2];		// 128
	//const int inputW = mInputDims.d[3];		// 128

	//// 验证一：是否是输入内存的问题
	//// 证明deviceBuffer.data()中的数据是正常的
	////memset(hostDataBuffer_half, 0, mInput->deviceBuffer.nbBytes());
	////CHECK(cudaMemcpy(hostDataBuffer_half, mInput->deviceBuffer.data(), mInput->deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));

	//// 验证二：是否是输出内存的问题
	//

	//// 分配临时内存，用于保存float32图像
	//float * imgOutTemp = new float[inputH*inputW];
	//for (int batch = 0; batch < mInputDims.d[0]; batch++)
	//{
	//	// fp16 保存到fp32中，然后存图
	//	if (mParams.fp16)
	//	{
	//		for (int i = 0; i < inputH*inputW; i++)
	//			imgOutTemp[i] = hostDataBuffer_half[batch*inputH*inputW + i];

	//		Mat imgNorm(inputH, inputW, CV_32FC1, imgOutTemp);
	//		if (!imwrite(mParams.inputNormFileName[batch], imgNorm))
	//		{
	//			cout << "imgNorm " << batch << ".tif保存失败" << endl;
	//		}
	//	}
	//	else
	//	{
	//		// 将数据保存为float32 然后保存
	//		Mat imgNorm(inputH, inputW, CV_32FC1, hostDataBuffer + batch * inputH*inputW);
	//		if (!imwrite(mParams.inputNormFileName[batch], imgNorm))
	//		{
	//			cout << "imgNorm " << batch << ".tif保存失败" << endl;
	//		}
	//	}
	//}
	//delete imgOutTemp;
	return true;
}

bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
 //   const int outputSize = mOutputDims.d[1];
 //   float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

	//const int outputH = mOutputDims.d[2];		// 1024
	//const int outputW = mOutputDims.d[3];		// 1024

	//Mat outputImg(outputH, outputW, CV_32FC1, output);

	//if (!imwrite("outputImg.tif", outputImg))
	//{
	//	cout << "保存结果失败" << endl;
	//}

 //   //float val{0.0f};
 //   //int idx{0};

 //   //// Calculate Softmax
 //   //float sum{0.0f};
 //   //for (int i = 0; i < outputSize; i++)
 //   //{
 //   //    output[i] = exp(output[i]);
 //   //    sum += output[i];
 //   //}

 //   //sample::gLogInfo << "Output:" << std::endl;
 //   //for (int i = 0; i < outputSize; i++)
 //   //{
 //   //    output[i] /= sum;
 //   //    val = std::max(val, output[i]);
 //   //    if (val == output[i])
 //   //    {
 //   //        idx = i;
 //   //    }

 //   //    sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
 //   //                     << " "
 //   //                     << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
 //   //                     << std::endl;
 //   //}
 //   //sample::gLogInfo << std::endl;

 //   //return idx == mNumber && val > 0.9f;

	return true;
}

// 函数重载
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::ManagedBuffer& buffers, TinyTIFFWriterFile* tif)
{
	//const int outputBatch = mOutputDims.d[0];
	//const int outputH = mOutputDims.d[2];		// 1024
	//const int outputW = mOutputDims.d[3];		// 1024

	//if (mParams.fp16)
	//{
	//	half* output = (half*)buffers.hostBuffer.data();

	//	// 分配临时内存，用于保存float32图像
	//	float * imgOutTemp = new float[outputH*outputW];

	//	for (int batch = 0; batch < mInputDims.d[0]; batch++)
	//	{

	//		for (int i = 0; i < outputH*outputW; i++)
	//			imgOutTemp[i] = output[batch*outputH*outputW + i];

	//		Mat imgNorm(outputH, outputW, CV_32FC1, imgOutTemp);
	//		if (!imwrite(mParams.outputFileName[batch], imgNorm))
	//		{
	//			cout << "output " << batch << ".tif保存失败" << endl;
	//			return false;
	//		}
	//	}
	//}
	//else
	//{
	//	// fp32
	//	float* output = (float*)buffers.hostBuffer.data();

	//	if (tif)
	//	{
	//		for (int batch = 0; batch < mInputDims.d[0]; batch++)
	//		{
	//			TinyTIFFWriter_writeImage(tif, output + batch * outputH*outputW);
	//		}	
	//	}

	//	//for (int batch = 0; batch < mInputDims.d[0]; batch++)
	//	//{
	//	//	Mat imgNorm(outputH, outputW, CV_32FC1, output + batch * outputH*outputW);
	//	//	if (!imwrite(mParams.outputFileName[batch], imgNorm))
	//	//	{
	//	//		cout << "output " << batch << ".tif保存失败" << endl;
	//	//		return false;
	//	//	}
	//	//}
	//}
	return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
bool SampleOnnxMNIST::initializeSampleParams(string inputDataDir,string outputDataDir, int scaleFactor,int modelType, std::mutex* pMutex, std::condition_variable* pCondVal, bool* pIsArrFull, bool* pIsProcessOver, bool* pIsDataTakeOff, bool fp16)
{
	this->pMutex			= pMutex;			// 互斥量
	this->pCondVal			= pCondVal;			// 条件变量
	this->pIsArrFull		= pIsArrFull;
	this->pIsProcessOver	= pIsProcessOver;	
	this->pIsDataTakeOff	= pIsDataTakeOff;	// 数据是否已经取走了 

	/*	初始化	*/

	mParams.inputDataDir = inputDataDir;
	mParams.outputDataDir = outputDataDir;

	mParams.scaleFactor	= scaleFactor;
	mParams.inputWidth	= modelType;
	mParams.inputHight	= modelType;
	mParams.outputHight	= mParams.inputHight*mParams.scaleFactor;
	mParams.outputWidth	= mParams.inputWidth*mParams.scaleFactor;
	mParams.onnxFileName= mParams.inputDataDir + "\\modelDynamic_ours_" + to_string(modelType) + "x" + to_string(modelType) + ".onnx";
	//mParams.fileDir		= "D:\\project\\Pro7-mEDSR-STORM\\data\\expriment\\data2\\test\\MMStack_Pos-1_metadata-" + to_string(modelType) + "x" + to_string(modelType) + "/temp1";		// file directory;
	mParams.fp16		= fp16;		// 设置fp16
	if(mParams.fp16)
		mParams.engine_file	= mParams.outputDataDir + "\\" + "denseDL_fp16.trt";
	else
		mParams.engine_file = mParams.outputDataDir + "\\" + "denseDL_fp32.trt";
	mParams.outputTiffName = mParams.outputDataDir + "\\" + "outputTiffName.tif";

	/*
	for (int fileIndex = 0; fileIndex < mParams.fileNums; fileIndex++)
	{
		mParams.inputFileName[fileIndex] = mParams.fileDir + "/" + to_string(fileIndex + 1) + ".tif";
		if (mParams.fp16)
		{
			mParams.inputNormFileName[fileIndex] = "./result/imgNorm_fp16_" + to_string(fileIndex + 1) + "_" + to_string(mParams.inputWidth) + ".tif";
			mParams.outputFileName[fileIndex] = "./result/output_fp16_" + to_string(fileIndex + 1) + "_" + to_string(mParams.outputWidth) + ".tif";
		}
		else
		{
			mParams.inputNormFileName[fileIndex] = "./result/imgNorm_fp32_" + to_string(fileIndex + 1) + "_" + to_string(mParams.inputWidth) + ".tif";
			mParams.outputFileName[fileIndex] = "./result/output_fp32_" + to_string(fileIndex + 1) + "_" + to_string(mParams.outputWidth) + ".tif";
		}
	}
	*/
	mParams.inputTensorNames.push_back("input");				// 输入名称
	mParams.outputTensorNames.push_back("output");				// 输出名称
	mParams.dlaCore = mParams.useDLACore;						// -1
	mParams.int8 = mParams.runInInt8;							// false
	//params.fp16 = args.run InFp16;							// false

	// 构建输出图像主机内存，设备内存
	mParams.renderImgOut_host = new float[mParams.outputHight*mParams.outputWidth];
	cudaMalloc((void**)&mParams.renderImgOut_device, mParams.outputHight*mParams.outputWidth * sizeof(float));

	cout << "initialize finished" << endl;
	return true;
}

bool SampleOnnxMNIST::deinit()
{
	delete[] mParams.renderImgOut_host;
	bool state = cudaFree(mParams.renderImgOut_device);
	return true;
}

/*
参考资料：
【1-sampleOnnxMNIST.cpp程序解读】：https://blog.csdn.net/yanggg1997/article/details/111587687
【2-sampleOnnxMNIST.cpp程序解读】：https://blog.csdn.net/Johnson_star/article/details/107692357
*/
int main_sampleOnnxMNIST(int argc, char** argv)
{
	// 多线程
	std::mutex* pMutex					= new std::mutex;								// 互斥量
	std::condition_variable* pCondVal	= new std::condition_variable;					// 条件变量
	bool isArrFull		= false;
	bool isProcessOver	= false;
	bool isDataTakeOff	= false;

	bool fp16 = false;

	int modelType = 256;
	int scaleFactor = 8;

	string inputDataDir = "D:\\project\\Pro7-mEDSR-STORM\\code\\inference\\cpp\\data";
	string outputDataDir = "D:\\project\\Pro7-mEDSR-STORM\\code\\inference\\cpp\\result";

    auto sampleTest = sample::gLogger.defineTest("my tensorRT", argc, argv);	// 定义一个日志类
    sample::gLogger.reportTestStart(sampleTest);								// 记录日志的开始

	// 【】参数解析
    SampleOnnxMNIST sample;									// 定义一个sample实例,匿名对象的赋值
	sample.initializeSampleParams(inputDataDir, outputDataDir, scaleFactor, modelType, pMutex, pCondVal, &isArrFull, &isProcessOver, &isDataTakeOff, fp16);	// 初始化参数				

	// 【】构建网络
    if (!sample.build())										
    {
        return sample::gLogger.reportFail(sampleTest);
    }

	// 【】推断
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

	system("pause");
    return sample::gLogger.reportPass(sampleTest);				// 结束
}

