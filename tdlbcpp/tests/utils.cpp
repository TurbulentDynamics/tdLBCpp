#include <string>
#include <algorithm>

#include "gtest/gtest.h"

#include "utils.hpp"

namespace TestUtils
{
    std::string random_string(size_t length)
    {
        auto randchar = []() -> char
        {
            const char charset[] =
                "0123456789"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz";
            const size_t max_index = (sizeof(charset) - 1);
            return charset[rand() % max_index];
        };
        std::string str(length, 0);
        std::generate_n(str.begin(), length, randchar);
        return str;
    }

    std::string getTempFilename(const std::string filename)
    {
#ifndef KEEP_TEMP_FILES
        return testing::TempDir() + "/" + filename;
#else
        return std::string("/tmp/") + filename;
#endif
    }
    void removeTempFile(const std::string filename)
    {
#ifndef KEEP_TEMP_FILES
        std::remove(filename.c_str());
#endif
    }
}
