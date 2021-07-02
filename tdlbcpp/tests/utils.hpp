#include <string>
#include <cstdlib>

//#define KEEP_TEMP_FILES

namespace TestUtils
{
    // Parameters for randomized tests
    const int randomStringLength = 400;
    const double randomDoubleLowerBound = -10000;
    const double randomDoubleUpperBound = 10000;

    const std::string temporaryFilenameSuffx = "_to_delete.json";

    std::string random_string(size_t length);
    std::string getTempFilename();
    void removeTempFile(const std::string filename);
}