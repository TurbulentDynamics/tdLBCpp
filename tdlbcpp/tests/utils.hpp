#include <string>
#include <cstdlib>

#define KEEP_TEMP_FILES

namespace TestUtils
{
    std::string random_string(size_t length);
    std::string getTempFilename(const std::string fileName);
    void removeTempFile(const std::string filename);
}