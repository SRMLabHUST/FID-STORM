
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
	// ��1������Builder(�����Ż�ģ��+����Engine)
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));  
    if (!builder)
    {
        return false;
    }

	if (builder->platformHasFastFp16())
		cout << "��ǰƽ̨֧��fp16" << endl;
	else
		cout << "��ǰƽ̨��֧��fp16" << endl;

	// ��1.1��ʹ��builder����Network
		// 1) ��������
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

	// ��1.2��ʹ��builder����config
		// 1) ѡ�����㾫��(FP16 or INT8)
		// 2) �Ż�ģ�� ��������Ч���㡢�۵���������������
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

	// ʹ�� FP16
	if (mParams.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}

	// ��3�������Ż������ļ�������������
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

	//// �µ�����ָ��
	//auto profileCalib = builder->createOptimizationProfile();
	//// We do not need to check the return of setDimension and setCalibrationProfile here as all dims are explicitly set
	//profileCalib->setDimensions(inputName, OptProfileSelector::kMIN, mInputDims);
	//profileCalib->setDimensions(inputName, OptProfileSelector::kOPT, mInputDims);
	//profileCalib->setDimensions(inputName, OptProfileSelector::kMAX, mInputDims);
	//config->setCalibrationProfile(profileCalib);

	// ��2������parser���˴���������onnx�ļ�
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

	// ��3���������� engine (constructed)
    auto constructed = constructNetwork(builder, network, config, parser);  // ���뺯��
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
	// builder�����ڷ���cuda��
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

	// �������л�����
	if (!isFileExists_ifstream(mParams.engine_file))		// trt�ļ�������
	{
		ofstream outfile(mParams.engine_file.c_str(), ios::out | ios::binary);		// �����ļ�
		if (!outfile.is_open())	// �ļ���ʧ��
		{
			fprintf(stderr, "fail to open file to write:%s\n", mParams.engine_file.c_str());
			return false;
		}

		// ��������
		SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
		if (!plan)
		{
			return false;
		}

		// �����ļ�
		fprintf(stdout, "allocate memory size:%d byte\n", plan->size());							// ���������С
		unsigned char * p = (unsigned char*)plan->data();
		outfile.write((char*)p, plan->size());
		outfile.close();
	}

	// �����ƶ����л���
    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

	// ȥ���л�cuda����
		// ע�⣺ȥ���л�����һ���ƶ����ݣ�ģ����Ҫ����һ�£��ж���Ĵ���ʱ�䣬��ˣ���һ�β�����ʱ���ϳ�����⼸�Σ����Կ��������ƶ�ʱ��ʱ������
	ifstream fin(mParams.engine_file,ios::in|ios::binary);
	string cached_engine = "";
	while (fin.peek() != EOF)
	{
		stringstream buffer;
		buffer << fin.rdbuf();						// ��infile�������е����ض��򵽱�׼���cout��
		cached_engine.append(buffer.str());
	}

	fin.close();

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

	// ��Ҫ����������������Tensor����״���е���
    //ASSERT(network->getNbInputs() == 1);					// ��֤����batchsizeΪ1
    //mInputDims = network->getInput(0)->getDimensions();		// ��ȡ����ά��
    //ASSERT(mInputDims.nbDims == 4);							// ��֤����ά��Ϊ4

    //ASSERT(network->getNbOutputs() == 1);					// ��֤���batchsizeΪ1
    //mOutputDims = network->getOutput(0)->getDimensions();	// ��ȡ���ά��
    //ASSERT(mOutputDims.nbDims == 2);						// ��֤���ά��Ϊ2

    return true;
}

bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
	//���� onnxfile
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
	// ��֤�ȷ����ڴ棬��ȡͼ��Ȼ����ִ���������򱨿�ָ���ڴ���ʴ���
	std::unique_lock<mutex> sbguard1(*this->pMutex);
	this->pCondVal->wait(sbguard1, [this] {
		if (*pIsSetimgRawToHostBuffer == false)			// δ����imgRaw�ڴ棬δ����imgRaw��HostBuffer
			return false;								// ����
		return true;
	});

	// ��������ִ��context
    // Create RAII buffer manager object 
		// 1) ��������
		// 2������ִ�е� ��context��
    samplesCommon::BufferManager buffers(mEngine);								// ���� BufferManager �� MangerBuffer
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context)	return false;

	ASSERT(mParams.inputTensorNames.size() == 1);

    // �������봦��
		// 1����ȡ����������䵽CPU���뻺��
		// 2��ִ��copyInputToDevice()���Զ���䵽GPU���뻺��ӿ�
	if (!processInput(buffers))	return false;

	// ���������ڴ�
	mInput->deviceBuffer.resize(mInputDims);		// �����豸�ڴ�

	// ��������ڴ����
	//mOutput->hostBuffer.resize(mOutputDims);
	mOutput->deviceBuffer.resize(mOutputDims);		// ����豸�ڴ�

	// ������
	// Set the input size for the preprocessor
	// ָ��context�����ά�ȣ�mPreprocessorContext��
	CHECK_RETURN_W_MSG(context->setBindingDimensions(0, mInputDims), false, "Invalid binding dimensions.");
	// We can only run inference once all dynamic input shapes have been specified.
	// ��飬�Ƿ����еĶ�̬���붼��ָ�����
	if (!context->allInputDimensionsSpecified()) return false;
	std::vector<void*> inferBindings = { mInput->deviceBuffer.data(), mOutput->deviceBuffer.data() };

	int curBatch = 1;
	//TinyTIFFWriterFile* tif = TinyTIFFWriter_open(mParams.outputTiffName.data(), 32, TinyTIFFWriter_Float, 0, mParams.outputWidth, mParams.outputHight, TinyTIFFWriter_Greyscale);
	string outputFileName = mParams.outputDataDir + "\\renderImg_"+to_string(mParams.inputHight)+"x"+ to_string(mParams.inputWidth) +"_.tif";
	TinyTIFFWriterFile* tif = TinyTIFFWriter_open(outputFileName.c_str(), 32, TinyTIFFWriter_Float, 0, mParams.outputWidth, mParams.outputHight, TinyTIFFWriter_Greyscale);

	sbguard1.unlock();

	// ���������ڴ� -�� �豸�ڴ�
	//while (!*pIsProcessOver)

	while (!*pIsProcessOver)
	{
		std::unique_lock<mutex> sbguard1(*this->pMutex);
		this->pCondVal->wait(sbguard1, [this] {
			if (*pIsArrFull == false)						// δ��
				return false;								// ����
			return true;
		});
		//cout << "threadID:" << std::this_thread::get_id() << "�õ�������" << endl;

		auto startTime = chrono::high_resolution_clock::now();	// ��ǰʱ��
		//Mat img(this->mParams.inputHight*this->mParams.batchSize, this->mParams.inputWidth, CV_32FC1, hostDataBuffer);
		//imwrite("img.tif", img);

		// ��֤��û��������������ȫΪ0����£�������Ƿ�����
		//memset(hostDataBuffer_half, 0, mInput->deviceBuffer.nbBytes());

		if(!mParams.fp16)
			CHECK(cudaMemcpy(mInput->deviceBuffer.data()/*void* dst*/, hostDataBuffer/*mInput->hostBuffer.data() void* src*/, mInput->deviceBuffer.nbBytes()/*size_t count*/, cudaMemcpyHostToDevice/*cudaMemcpyKind kind*/));
		else
			CHECK(cudaMemcpy(mInput->deviceBuffer.data()/*void* dst*/, hostDataBuffer_half/*mInput->hostBuffer.data() void* src*/, mInput->deviceBuffer.nbBytes()/*size_t count*/, cudaMemcpyHostToDevice/*cudaMemcpyKind kind*/));

		//cout << _msize(hostDataBuffer_half) << endl;
		//cout << "threadID:" << std::this_thread::get_id() << "���ݿ�������" << endl;
		
		*pIsArrFull		= false;			// ����Ϊ����
		*pIsDataTakeOff = true;				// ����������
		this->pCondVal->notify_one();		// ֪ͨ�Ѿ�ȡ��������
		//cout << "threadID:" << std::this_thread::get_id() << "��ǰ֡��"<< this->dataloader.curFrame << endl;
		//cout << "threadID:" << std::this_thread::get_id() << "֪ͨ�⿪����״̬��" << endl;

		//������һ��ͼ��ı��棬������
		//normImgSave();
		sbguard1.unlock();
		
		//cout << "���host��buffer��С��"   << mOutput->hostBuffer.nbBytes() << endl;
		//cout << "���device��buffer��С��" << mOutput->deviceBuffer.nbBytes() << endl;

		//normImgSaveOutput();

		// ִ��
		bool status = context->executeV2(inferBindings.data());
		if (!status)
		{
			system("pause");
			return false;
		}

		// �����ó����
		//imgAdd <<<blockNums, threadPerBlock>>> (mOutput->deviceBuffer.data(), mParams.renderImgOut_device, mParams.outputWidth , mParams.outputHight, mParams.batchSize);
		imgAddTop((float*)mOutput->deviceBuffer.data(), mParams.renderImgOut_device, mParams.outputWidth , mParams.outputHight, mParams.batchSize,mParams.fp16);
		//CHECK(cudaMemcpy(
		//	mOutput->hostBuffer.data()/*void* dst*/, mOutput->deviceBuffer.data()/*void* src*/, mOutput->hostBuffer.nbBytes()/*size_t count*/, cudaMemcpyDeviceToHost/*cudaMemcpyKind kind*/));

		auto  endTime = chrono::high_resolution_clock::now();
		float totalTime = chrono::duration<float, milli>(endTime - startTime).count();

		cout << "Execution time:" << totalTime << "ms, current batch:" << curBatch++ << endl;
		//cout<< "mOutput��С��"<<_msize(mOutput->hostBuffer.data())<<endl;  // ��ȷ
		//cout << "���豸�������ڴ濽�����"  << endl;
		// ����ƶ�ͼ��
		//half * p = (half*)mOutput->hostBuffer.data();
		//for (int i = 0; i < 1024*10; i++)
		//	cout << p[i] << endl;

		// ����ͼ�񱣴�
		/*if (!verifyOutput(*mOutput,tif))	return false;*/
		//cout << "���ͼ�񱣴����" << endl;
	}
	
	cudaError_t state = cudaMemcpy(mParams.renderImgOut_host, mParams.renderImgOut_device, mParams.outputWidth * mParams.outputHight * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Is output image error?:" << state << endl;
	TinyTIFFWriter_writeImage(tif, mParams.renderImgOut_host);
	TinyTIFFWriter_close(tif);			// �رյ�ǰͼ��

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[2];		// 128
    const int inputW = mInputDims.d[3];		// 128

	// ��������host buffer
	if (mParams.fp16)
	{
		// ����
		mInput->hostBuffer		= samplesCommon::HostBuffer(nvinfer1::DataType::kHALF);
		mInput->deviceBuffer	= samplesCommon::DeviceBuffer(nvinfer1::DataType::kHALF);
		// ���
		mOutput->hostBuffer		= samplesCommon::HostBuffer(nvinfer1::DataType::kHALF);
		mOutput->deviceBuffer	= samplesCommon::DeviceBuffer(nvinfer1::DataType::kHALF);
	}
	
	if (mParams.fp16)
	{
		hostDataBuffer_half = this->dataloader.imgRaw_fp16;
		//cout << "����ͼ�����������ڴ棺" << _msize(hostDataBuffer_half) << endl;
	}
	else
		hostDataBuffer = this->dataloader.imgRaw;

    return true;
}

bool SampleOnnxMNIST::normImgSaveOutput()
{
	//const int OutputH = mOutputDims.d[2];		// 128
	//const int OutputW = mOutputDims.d[3];		// 128

	//// ��֤һ���Ƿ��������ڴ������
	//// ֤��deviceBuffer.data()�е�������������
	//half* outHostDataBuffer_half = (half*)mOutput->hostBuffer.data();
	//memset(outHostDataBuffer_half, 0, mOutput->hostBuffer.nbBytes());
	//CHECK(cudaMemcpy(outHostDataBuffer_half, mOutput->deviceBuffer.data(), mOutput->deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));	
	//
	//// ������ʱ�ڴ棬���ڱ���float32ͼ��
	//float * imgOutTemp = new float[OutputH*OutputW];
	//for (int batch = 0; batch < mOutputDims.d[0]; batch++)
	//{
	//	// fp16 ���浽fp32�У�Ȼ���ͼ
	//	if (mParams.fp16)
	//	{
	//		for (int i = 0; i < OutputW*OutputH; i++)
	//			imgOutTemp[i] = outHostDataBuffer_half[batch*OutputW*OutputH + i];

	//		Mat imgNorm(OutputH, OutputW, CV_32FC1, imgOutTemp);
	//		if (!imwrite(mParams.outputFileName[batch], imgNorm))
	//		{
	//			cout << "imgNorm " << batch << ".tif����ʧ��" << endl;
	//		}
	//	}
	//	else
	//	{
	//		// �����ݱ���Ϊfloat32 Ȼ�󱣴�
	//		Mat imgNorm(OutputH, OutputW, CV_32FC1, outHostDataBuffer_half + batch * OutputH*OutputW);
	//		if (!imwrite(mParams.outputFileName[batch], imgNorm))
	//		{
	//			cout << "imgNorm " << batch << ".tif����ʧ��" << endl;
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

	//// ��֤һ���Ƿ��������ڴ������
	//// ֤��deviceBuffer.data()�е�������������
	////memset(hostDataBuffer_half, 0, mInput->deviceBuffer.nbBytes());
	////CHECK(cudaMemcpy(hostDataBuffer_half, mInput->deviceBuffer.data(), mInput->deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));

	//// ��֤�����Ƿ�������ڴ������
	//

	//// ������ʱ�ڴ棬���ڱ���float32ͼ��
	//float * imgOutTemp = new float[inputH*inputW];
	//for (int batch = 0; batch < mInputDims.d[0]; batch++)
	//{
	//	// fp16 ���浽fp32�У�Ȼ���ͼ
	//	if (mParams.fp16)
	//	{
	//		for (int i = 0; i < inputH*inputW; i++)
	//			imgOutTemp[i] = hostDataBuffer_half[batch*inputH*inputW + i];

	//		Mat imgNorm(inputH, inputW, CV_32FC1, imgOutTemp);
	//		if (!imwrite(mParams.inputNormFileName[batch], imgNorm))
	//		{
	//			cout << "imgNorm " << batch << ".tif����ʧ��" << endl;
	//		}
	//	}
	//	else
	//	{
	//		// �����ݱ���Ϊfloat32 Ȼ�󱣴�
	//		Mat imgNorm(inputH, inputW, CV_32FC1, hostDataBuffer + batch * inputH*inputW);
	//		if (!imwrite(mParams.inputNormFileName[batch], imgNorm))
	//		{
	//			cout << "imgNorm " << batch << ".tif����ʧ��" << endl;
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
	//	cout << "������ʧ��" << endl;
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

// ��������
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::ManagedBuffer& buffers, TinyTIFFWriterFile* tif)
{
	//const int outputBatch = mOutputDims.d[0];
	//const int outputH = mOutputDims.d[2];		// 1024
	//const int outputW = mOutputDims.d[3];		// 1024

	//if (mParams.fp16)
	//{
	//	half* output = (half*)buffers.hostBuffer.data();

	//	// ������ʱ�ڴ棬���ڱ���float32ͼ��
	//	float * imgOutTemp = new float[outputH*outputW];

	//	for (int batch = 0; batch < mInputDims.d[0]; batch++)
	//	{

	//		for (int i = 0; i < outputH*outputW; i++)
	//			imgOutTemp[i] = output[batch*outputH*outputW + i];

	//		Mat imgNorm(outputH, outputW, CV_32FC1, imgOutTemp);
	//		if (!imwrite(mParams.outputFileName[batch], imgNorm))
	//		{
	//			cout << "output " << batch << ".tif����ʧ��" << endl;
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
	//	//		cout << "output " << batch << ".tif����ʧ��" << endl;
	//	//		return false;
	//	//	}
	//	//}
	//}
	return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
bool SampleOnnxMNIST::initializeSampleParams(string inputDataDir, string outputDataDir, int scaleFactor, int modelType, std::mutex* pMutex, std::condition_variable* pCondVal, bool* pIsArrFull, bool* pIsProcessOver, bool* pIsDataTakeOff, bool* pIsSetimgRawToHostBuffer, bool fp16)
{
	this->pMutex			= pMutex;			// ������
	this->pCondVal			= pCondVal;			// ��������
	this->pIsArrFull		= pIsArrFull;
	this->pIsSetimgRawToHostBuffer = pIsSetimgRawToHostBuffer;
	this->pIsProcessOver	= pIsProcessOver;	
	this->pIsDataTakeOff	= pIsDataTakeOff;	// �����Ƿ��Ѿ�ȡ���� 

	/*	��ʼ��	*/
	mParams.inputDataDir = inputDataDir;
	mParams.outputDataDir = outputDataDir;

	mParams.batchSize	= this->dataloader.batchSize;
	mParams.scaleFactor	= scaleFactor;
	mParams.inputWidth	= modelType;
	mParams.inputHight	= modelType;
	mParams.outputHight	= mParams.inputHight*mParams.scaleFactor;
	mParams.outputWidth	= mParams.inputWidth*mParams.scaleFactor;
	mParams.onnxFileName= mParams.inputDataDir + "\\modelDynamic_ours_" + to_string(modelType) + "x" + to_string(modelType) + ".onnx";
	//mParams.fileDir		= "D:\\project\\Pro7-mEDSR-STORM\\data\\expriment\\data2\\test\\MMStack_Pos-1_metadata-" + to_string(modelType) + "x" + to_string(modelType) + "/temp1";		// file directory;
	mParams.fp16		= fp16;		// ����fp16
	if(mParams.fp16)
		mParams.engine_file	= mParams.outputDataDir + "\\" + "denseDL_fp16_" + to_string(modelType) + "x" + to_string(modelType) + "_batchsize_" + to_string(mParams.batchSize) + ".trt";
	else
		mParams.engine_file = mParams.outputDataDir + "\\" + "denseDL_fp32_" + to_string(modelType) + "x" + to_string(modelType) +"_batchsize_"+to_string(mParams.batchSize)+ ".trt";
	mParams.outputTiffName = mParams.outputDataDir + "\\" + "rawImg_" + to_string(modelType) + "x" + to_string(modelType) + "_output.tif";

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
	mParams.inputTensorNames.push_back("input");				// ��������
	mParams.outputTensorNames.push_back("output");				// �������
	mParams.dlaCore = mParams.useDLACore;						// -1
	mParams.int8 = mParams.runInInt8;							// false
	//params.fp16 = args.run InFp16;							// false

	// �������ͼ�������ڴ棬�豸�ڴ�
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
�ο����ϣ�
��1-sampleOnnxMNIST.cpp����������https://blog.csdn.net/yanggg1997/article/details/111587687
��2-sampleOnnxMNIST.cpp����������https://blog.csdn.net/Johnson_star/article/details/107692357
*/
int main_sampleOnnxMNIST(int argc, char** argv)
{
	// ���߳�
	std::mutex* pMutex					= new std::mutex;								// ������
	std::condition_variable* pCondVal	= new std::condition_variable;					// ��������
	bool isArrFull		= false;
	bool isProcessOver	= false;
	bool isDataTakeOff	= false;
	bool pIsSetimgRawToHostBuffer = false;


	bool fp16 = false;

	int modelType = 256;
	int scaleFactor = 8;

	string inputDataDir = "D:\\project\\Pro7-mEDSR-STORM\\code\\inference\\cpp\\data";
	string outputDataDir = "D:\\project\\Pro7-mEDSR-STORM\\code\\inference\\cpp\\result";

    auto sampleTest = sample::gLogger.defineTest("my tensorRT", argc, argv);	// ����һ����־��
    sample::gLogger.reportTestStart(sampleTest);								// ��¼��־�Ŀ�ʼ

	// ������������
    SampleOnnxMNIST sample;									// ����һ��sampleʵ��,��������ĸ�ֵ
	sample.initializeSampleParams(inputDataDir, outputDataDir, scaleFactor, modelType, pMutex, pCondVal, &isArrFull, &isProcessOver, &isDataTakeOff, &pIsSetimgRawToHostBuffer, fp16);	// ��ʼ������				

	// ������������
    if (!sample.build())										
    {
        return sample::gLogger.reportFail(sampleTest);
    }

	// �����ƶ�
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

	system("pause");
    return sample::gLogger.reportPass(sampleTest);				// ����
}

