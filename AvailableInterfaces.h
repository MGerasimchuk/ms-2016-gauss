/** FOR IMAGE HELPER */
#include <map>
#include "ImageInterface.h"
#include "PPMInterface.h"
#include "StringHelper.h"


/** ALLOW EXTENSIONS */
std::map<std::string, ImageInterface*> helpers = {
	{ "ppm", new PPMInterface() } //PPM
};
