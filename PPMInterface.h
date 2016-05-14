#include "ImageInterface.h"

class PPMInterface : public ImageInterface
{
public:
	virtual unsigned char ** load(const char *file,
		unsigned int *w, unsigned int *h);
};