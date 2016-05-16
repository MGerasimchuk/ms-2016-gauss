#include "PPMInterface.h"
#include "helper_image.h"

unsigned char ** PPMInterface::load(const char *file,
	unsigned int *w, unsigned int *h)
{
	unsigned int *temp = NULL;
	sdkLoadPPM4ub(file, (unsigned char **)&temp, w, h);
	return (unsigned char **)temp;
}

bool PPMInterface::save(const char *file, unsigned char *data,
	unsigned int w, unsigned int h)
{
	return sdkSavePPM4ub(file, data, w, h);
}

int PPMInterface::getSizeof()
{
	return sizeof(unsigned int);
}