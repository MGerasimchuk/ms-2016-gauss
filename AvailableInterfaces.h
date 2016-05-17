/** FOR IMAGE HELPER */
#include <map>
#include "ImageInterface.h"
#include "PPMInterface.h"
#include "StringHelper.h"
#include "PGMInterface.h"


/** ALLOW EXTENSIONS */
std::map<std::string, ImageInterface*> helpers = {
	{ "ppm", new PPMInterface() }, //PPM
	{ "pgm", new PGMInterface() }, //PGM
};
