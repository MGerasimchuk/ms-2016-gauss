#pragma warning(disable:4819)

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

const char *sSDKsample = "CUDA Recursive Gaussian";

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
	sdkSavePPM4ub(dump_file, h_result, width, height);

	if (!sdkComparePPM(dump_file, sdkFindFilePath(ref_file, exec_path), MAX_EPSILON_ERROR, THRESHOLD, false))
	{
		nTotalErrors++;
	}

	printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
	printf("%.2f Mpixels/sec\n", (width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);

	checkCudaErrors(cudaFree(d_result));
	free(h_result);

	printf("Summary: %d errors!\n", nTotalErrors);

	printf(nTotalErrors == 0 ? "Test passed\n" : "Test failed!\n");
	return (nTotalErrors == 0);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
	printf("%s Starting...\n\n", sSDKsample);

	//printf("NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

	// Get the path of the filename
	char *image_path = NULL;
	image_path = new char[255];
	strcpy(image_path, "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v7.5/3_Imaging/recursiveGaussian/data/lena.ppm");

	sdkLoadPPM4ub(image_path, (unsigned char **)&h_img, &width, &height);

	if (!h_img)
	{
		printf("Error unable to load PPM file: '%s'\n", image_path);
		exit(EXIT_FAILURE);
	}

	printf("Loaded '%s', %d x %d pixels\n", image_path, width, height);

	nthreads = 64; //потоки
	sigma = 1; //радиус размытия


	initCudaBuffers();

	if (image_path)
	{
		printf("(Automated Testing)\n");
		bool testPassed = runSingleTest(image_path, argv[0]);

		cleanup();

		// cudaDeviceReset causes the driver to clean up all state. While
		// not mandatory in normal operation, it is good practice.  It is also
		// needed to ensure correct operation when the application is being
		// profiled. Calling cudaDeviceReset causes all profile data to be
		// flushed before the application exits
		cudaDeviceReset();

		exit(testPassed ? EXIT_SUCCESS : EXIT_FAILURE);
	}
}
