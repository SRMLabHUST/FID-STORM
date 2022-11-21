//��1-����sampleOnnxMNIST��tensorrt�޸ġ� https://github.com/ILoveU3D/tensorrt

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
//	engineFile.seekg(0, ifstream::end);		// ���ô�����������ȡ���¸��ַ���λ��
//	int64_t fsize = engineFile.tellg();		// ���ָ�������е�ǰλ��
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
//// �������ݴ���
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
//// �������Ľ���
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
//�ο����ϣ�
//1�� mobilenet TensorRT demo : https://github.com/ILoveU3D/tensorrt
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

	// ����
	string onnxFileName = "D:/project/Pro7-denseDL/code/denseDL/onnxConvert/model_ours.onnx";

	// ����builder
	IBuilder * builder = createInferBuilder(gLogger);
	
	// ����network
	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

	// �����ض���ʽ�Ľ����� ��onnx��
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

	// ���� onnxfile
	parser->parseFromFile(onnxFileName.c_str(),static_cast<int>(gLogger.getReportableSeverity()));
	for (int i = 0; i < parser->getNbErrors(); ++i)
	{
		cout << parser->getError(i)->desc() << std::endl;
	}
	

	// �������� (��������ʱ��TensorRT�Ḵ��Ȩ��)
	IBuilderConfig* config = builder->createBuilderConfig();
	config->setMaxWorkspaceSize(1 << 20);
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	
	// ����
	// �����ռ����洢������ִ����Ҫ�Ķ���洢�ռ�
	IExecutionContext * context = engine->createExecutionContext();

	int inputIndex  = engine->getBindingIndex("input");
	int outputIndex = engine->getBindingIndex("output");

	//void * buffers[2];
	//buffers[inputIndex] = inputbu






	system("pause");
	return;

}