//【1-基于sampleOnnxMNIST的tensorrt修改】 https://github.com/ILoveU3D/tensorrt

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <ctime>

using namespace samplesCommon;
using namespace sample;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;

//class _declspec(dllexport) OnnxMobileNet
//{
//private:
//	OnnxSampleParams params;
//	const int inputWeight;
//	const int inputHeight;
//	const int OutputLength;
//
//	shared_ptr<ICudaEngine> engine;
//	SampleUniquePtr<IExecutionContext> context;
//	void processInput(BufferManager& buffers, string imgName);
//	int64_t verifyOutput(BufferManager& buffers);
//public:
//	OnnxMobileNet(const OnnxSampleParams& params, const int inputWeight, const int inputHeight, const int Outputlength) :params(params), inputWeight(inputWeight), inputHeight(inputHeight), OutputLength(OutputLength) {}
//	void build(string engineName);
//	int64_t infer(string imgName);
//};
//
//void OnnxMobileNet::build(string engineName)
//{
//	ifstream engineFile(engineName, ios::binary);
//	assert(engineFile);
//
//	engineFile.seekg(0, ifstream::end);		// 设置从输入流中提取的下个字符的位置
//	int64_t fsize = engineFile.tellg();		// 获得指针在流中当前位置
//	engineFile.seekg(0, ifstream::beg);		// 
//
//	vector<char> engineData(fsize);
//	engineFile.read(engineData.data(), fsize);
//	assert(engineFile);
//
//	SampleUniquePtr<IRuntime> runtime = SampleUniquePtr<IRuntime>(createInferRuntime(gLogger.getTRTLogger()));
//	assert(runtime);
//
//	ICudaEngine* engineTRT = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
//	assert(engineTRT);
//
//	engine = shared_ptr<ICudaEngine>(engineTRT, InferDeleter());
//	assert(engine);
//
//	context = SampleUniquePtr<IExecutionContext>(engine->createExecutionContext());
//	assert(context);
//}
//
//int64_t OnnxMobileNet::infer(string imgName) {
//
//	BufferManager buffers(engine);
//
//	processInput(buffers, imgName);
//
//	buffers.copyInputToDevice();
//
//	bool status = context->executeV2(buffers.getDeviceBindings().data());
//	assert(status);
//
//	buffers.copyOutputToHost();
//
//	return verifyOutput(buffers);
//}
//
//// 输入数据处理
//void OnnxMobileNet::processInput(BufferManager& buffers, string imgName) {
//	string fileName = locateFile(imgName, params.dataDirs);
//	Mat img = imread(fileName, IMREAD_GRAYSCALE);
//
//	//imshow("fileName", img);
//	//waitKey(0);
//
//	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(params.inputTensorNames[0]));
//	for (int i = 0; i < inputWeight*inputHeight; i++) {
//		hostDataBuffer[i] = float(img.data[i]);
//	}
//}
//
//// 输出结果的解析
//int64_t OnnxMobileNet::verifyOutput(BufferManager& buffers) {
//	const int outputLength = 30;
//	float* output = static_cast<float*>(buffers.getHostBuffer(params.outputTensorNames[0]));
//	int64_t maxIdx = 0;
//
//	for (int64_t i = 1; i < outputLength; i++) {
//		if (output[maxIdx] < output[i])
//			maxIdx = i;
//	}
//
//	return maxIdx;
//}
//
//
///*
//参考资料：
//1） mobilenet TensorRT demo : https://github.com/ILoveU3D/tensorrt
//*/
//int main01(int argc, char **argv) {
//	OnnxSampleParams params;
//	params.dataDirs.push_back(".\\");
//	params.dataDirs.push_back(".\\ref\\testData");
//	params.onnxFileName = "model.onnx";
//	params.inputTensorNames.push_back("input:0");
//	params.outputTensorNames.push_back("Reshape_1:0");
//	OnnxMobileNet model(params, 32, 32, 30);
//	model.build("mobilenet_v1_1.0_224.trt");
//
//	ifstream filenames(".\\ref\\filenames.txt");
//	cout << filenames.is_open() << endl;
//	string filename;
//
//	clock_t start = clock();
//	int64_t i = 1;
//
//	while (getline(filenames, filename)) {
//		cout << filename << ends;
//		cout << model.infer(filename) << endl;
//		i++;
//	}
//	filenames.close();
//
//	clock_t finish = clock();
//	cout << "total time = " << ((finish - start) * 1000) / CLOCKS_PER_SEC << endl;
//	cout << "total account = " << i << endl;
//
//	return 0;
//}

void main001()
{

	// 参数
	string onnxFileName = "D:/project/Pro7-denseDL/code/denseDL/onnxConvert/model_ours.onnx";

	// 创建builder
	IBuilder * builder = createInferBuilder(gLogger);
	
	// 创建network
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

	// 创建特定格式的解析器 （onnx）
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

	// 解析 onnxfile
	parser->parseFromFile(onnxFileName.c_str(),static_cast<int>(gLogger.getReportableSeverity()));
	for (int i = 0; i < parser->getNbErrors(); ++i)
	{
		cout << parser->getError(i)->desc() << std::endl;
	}
	

	// 构建引擎 (构建引擎时，TensorRT会复制权重)
	IBuilderConfig* config = builder->createBuilderConfig();
	config->setMaxWorkspaceSize(1 << 20);
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	
	// 推理
	// 创建空间来存储上下文执行需要的额外存储空间
	IExecutionContext * context = engine->createExecutionContext();

	int inputIndex  = engine->getBindingIndex("input");
	int outputIndex = engine->getBindingIndex("output");

	//void * buffers[2];
	//buffers[inputIndex] = inputbu






	system("pause");
	return;

}