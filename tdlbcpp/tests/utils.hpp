#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <string>
#include <cstdlib>

namespace TestUtils
{
    // Parameters for randomized tests
    const int randomStringLength = 400;
    const double randomDoubleLowerBound = -10000;
    const double randomDoubleUpperBound = 10000;
    const int randomArraySize = 5;
    const int randomArrayMinimalSize = 2;

    const std::string temporaryFilenameSuffix = "_to_delete.json";

    std::string random_string(size_t length);
    std::string getTempFilename();
    std::string getTempFilename(std::string folder, std::string suffix);
    void removeTempFile(const std::string filename);
    void captureStderr();
    std::string getCapturedStderr();
    std::string joinPath(const std::string &p1, const std::string &p2);
    void createDir(const std::string &path);
}

#endif //UTILS_H_INCLUDED
