#include "PGMInterface.h"
#include "helper_image.h"

unsigned char ** PGMInterface::load(const char *file,
	unsigned int *w, unsigned int *h)
{
	unsigned int *temp = NULL;
	sdkLoadPGM(file, (unsigned char **)&temp, w, h);
	return (unsigned char **)temp;
}

bool PGMInterface::save(const char *file, unsigned char *data,
	unsigned int w, unsigned int h)
{
	return sdkSavePGM(file, data, w, h);
}

int PGMInterface::getSizeof()
{
	return sizeof(unsigned char);
}