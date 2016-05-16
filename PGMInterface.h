#include "ImageInterface.h"

class PGMInterface : public ImageInterface
{
public:
	virtual unsigned char ** load(const char *file,
		unsigned int *w, unsigned int *h);

	virtual bool save(const char *file, unsigned char *data,
		unsigned int w, unsigned int h);

	virtual int getSizeof();
};