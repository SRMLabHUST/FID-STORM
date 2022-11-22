#include "dll.h"
#include "main00.h"
#include <iostream>
#include <windows.h>
using namespace std;

//Hardware Initialization

JNIEXPORT jint JNICALL Java_dll_Initialization
(JNIEnv *env, jobject obj, jstring jinputDataDir, jstring joutputDataDir, jdouble jbatchSize, jdouble jmodelType, jdouble jAmplification , jboolean jfp)
{
	// Open the console
	AllocConsole();
	freopen("CONOUT$", "a+", stdout);


	int argc = 0;
	char** argv = nullptr;
	const char* jF = env->GetStringUTFChars(jinputDataDir, NULL);
	if (jF==NULL)
	{
		return -1;
	}
	string inputDataDir(jF);
	const char* jM = env->GetStringUTFChars(joutputDataDir, NULL);
	if (jM == NULL)
	{
		return -1;
	}
	string outputDataDir(jM);
	int batchSize = (int)jbatchSize;
	int modelType = (int)jmodelType;
	int Amplification = (int)jAmplification;
	bool fp = jfp;
	main00(argc, argv, inputDataDir, outputDataDir, batchSize, modelType, Amplification, fp);

	//Release pointer jF¡¢jM
	env->ReleaseStringUTFChars(jinputDataDir, jF);
	env->ReleaseStringUTFChars(joutputDataDir, jM);

	return 1000;
}
