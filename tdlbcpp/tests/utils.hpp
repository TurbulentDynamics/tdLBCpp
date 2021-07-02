#include <string>
#include <cstdlib>

//#define KEEP_TEMP_FILES

namespace TestUtils
{
    // Parameters for randomized tests
    const int randomStringLength = 400;
    const double randomDoubleLowerBound = -10000;
    const double randomDoubleUpperBound = 10000;

    std::string random_string(size_t length);
    std::string getTempFilename(const std::string fileName);
    void removeTempFile(const std::string filename);
}