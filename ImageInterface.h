#ifndef __FACTORY__
#define __FACTORY__

class ImageInterface
{
public:
	virtual bool read(const char *file, unsigned char **data,
		unsigned int *w, unsigned int *h, unsigned int *channels) = 0;
};

#endif