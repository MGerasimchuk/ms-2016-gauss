#include "PPMInterface.h"
#include "helper_image.h"

bool PPMInterface::load(const char *file, unsigned char **data,
	unsigned int *w, unsigned int *h)
{
	sdkLoadPPM4ub(file, (unsigned char **)&data, w, h);
}