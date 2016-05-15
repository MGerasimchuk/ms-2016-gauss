﻿#pragma warning(disable:4819)

// CUDA includes and interop headers
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include "helper_functions.h"
#include "helper_cuda.h"      // includes cuda.h and cuda_runtime_api.h

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/** FOR IMAGE HELPER */
#include <map>
#include "ImageInterface.h"
#include "PPMInterface.h"
#include "StringHelper.h"



#define MAX(a,b) ((a > b) ? a : b)

#define USE_SIMPLE_FILTER 0

#define MAX_EPSILON_ERROR 5.0f
#define THRESHOLD  0.15f

float sigma = 10.0f;
int order = 0;
int nthreads = 64;  // number of threads per block

unsigned int width, height;
unsigned int *h_img = NULL;
unsigned int *d_img = NULL;
unsigned int *d_temp = NULL;

StopWatchInterface *timer = 0;

bool runBenchmark = false;
bool printTimings = false;

/** ALLOW EXTENSIONS */
std::map<std::string, ImageInterface*> helpers = {
	{ "ppm", new PPMInterface() } //PPM
};

extern "C"
void transpose(unsigned int *d_src, unsigned int *d_dest, unsigned int width, int height);

extern "C"
void gaussianFilterRGBA(unsigned int *d_src, unsigned int *d_dest, unsigned int *d_temp, int width, int height, float sigma, int order, int nthreads);

void cleanup();

void cleanup()
{
	sdkDeleteTimer(&timer);

	checkCudaErrors(cudaFree(d_img));
	checkCudaErrors(cudaFree(d_temp));

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
}

void initCudaBuffers()
{
	unsigned int size = width * height * sizeof(unsigned int);

	// allocate device memory
	checkCudaErrors(cudaMalloc((void **)&d_img, size));
	checkCudaErrors(cudaMalloc((void **)&d_temp, size));

	checkCudaErrors(cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice));

	sdkCreateTimer(&timer);
}

bool
runSingleTest(const char *ref_file, const char *exec_path)
{
	// allocate memory for result
	int nTotalErrors = 0;
	unsigned int *d_result;
	unsigned int size = width * height * sizeof(unsigned int);
	checkCudaErrors(cudaMalloc((void **)&d_result, size));

	// warm-up
	gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, order, nthreads);

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStartTimer(&timer);

	gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, order, nthreads);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("Kernel execution failed");
	sdkStopTimer(&timer);

	unsigned char *h_result = (unsigned char *)malloc(width*height * 4);
	checkCudaErrors(cudaMemcpy(h_result, d_result, width*height * 4, cudaMemcpyDeviceToHost));

	char dump_file[1024];
	sprintf(dump_file, "lena_%02d.ppm", (int)sigma);

	std::string ext = getExtension(dump_file);
	helpers[ext]->save(dump_file, h_result, width, height);

	//sdkSavePPM4ub(dump_file, h_result, width, height);

	/*if (!sdkComparePPM(dump_file, sdkFindFilePath(ref_file, exec_path), MAX_EPSILON_ERROR, THRESHOLD, false))
	{
		nTotalErrors++;
	}*/

	if (printTimings) {
		printf("  Finished %f sec\n", sdkGetTimerValue(&timer) / 1000.0);
	}
	
	//printf("%.2f Mpixels/sec\n", (width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);

	//checkCudaErrors(cudaFree(d_result));
	free(h_result);

	//printf("Summary: %d errors!\n", nTotalErrors);

	//printf(nTotalErrors == 0 ? "Test passed\n" : "Test failed!\n");
	return (nTotalErrors == 0);
}


void applyFilter(const char * image_path, const char * outputFile)
{
	//printf("Starting...\n\n");
	printf("- Processing Gaussian blur on %s...\n", image_path);

	std::string ext = getExtension(image_path);

	if (helpers.count(ext) == 0) {
		std::cout << "  Format *." << ext << " is not supported.\n";
		return;
	}

	unsigned char ** temp = helpers[ext]->load(image_path, &width, &height);
	h_img = (unsigned int*)temp;
	//sdkLoadPPM4ub(image_path, (unsigned char **)&h_img, &width, &height);

	if (!h_img)
	{
		//printf("Error unable to load file: '%s'\n", image_path);
		exit(EXIT_FAILURE);
	}

	//printf("Loaded '%s', %d x %d pixels\n", image_path, width, height);
	

	nthreads = 64; //потоки
	//sigma = 1; //радиус размытия


	initCudaBuffers();

	if (image_path)
	{
		//printf("(Automated Testing)\n");
		bool testPassed = runSingleTest(image_path, outputFile);

		cleanup();

		// cudaDeviceReset causes the driver to clean up all state. While
		// not mandatory in normal operation, it is good practice.  It is also
		// needed to ensure correct operation when the application is being
		// profiled. Calling cudaDeviceReset causes all profile data to be
		// flushed before the application exits
		cudaDeviceReset();
	}
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	if (argc < 2) {
		printf("Error input arguments!\n");
		return 0;
	}

	int startArgIndex = 1;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-t") == 0) {
			printTimings = true;

			startArgIndex = i + 1;
		}

		if (strcmp(argv[i], "-r") == 0 && (i + 1) < argc) {
			printTimings = true;
			// Get the path of the filename
			sigma = atoi(argv[i + 1]); //читаем радиус размытия из входных параметров
			startArgIndex = i + 2;
			break;
		}
		
	}
	
	/** Обрабатывем файлы*/
	for (int i = startArgIndex; i < argc; i++) {
		applyFilter(argv[i], argv[0]);
	}
}