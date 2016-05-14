#include "ImageInterface.h"

class PPMInterface : public ImageInterface
{
public:
	virtual bool load(const char *file, unsigned char **data,
		unsigned int *w, unsigned int *h);
};