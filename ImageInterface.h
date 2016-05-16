#ifndef __FACTORY__
#define __FACTORY__

class ImageInterface
{
public:
	virtual unsigned char ** load(const char *file,
		unsigned int *w, unsigned int *h) = 0;

	virtual bool save(const char *file, unsigned char *data,
		unsigned int w, unsigned int h) = 0;

	virtual int getSizeof() = 0;
};

#endif