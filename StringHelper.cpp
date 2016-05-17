#include "StringHelper.h"


std::string getExtension(std::string filename)
{
	std::size_t found = filename.find_last_of('.') + 1;
	std::string ext = filename.substr(found, filename.length());
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	return ext;
}