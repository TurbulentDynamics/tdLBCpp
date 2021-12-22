#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include <string>

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

    std::string getTestName()
    {
        const testing::TestInfo *const test_info =
            testing::UnitTest::GetInstance()->current_test_info();
        return test_info->name();
    }

    std::string getTempFilename(std::string folder, std::string suffix)
    {
        std::string testName = getTestName();
        return folder + "/" + testName + suffix;
    }

    std::string getTempFilename(std::string folder)
    {
        return getTempFilename(folder, temporaryFilenameSuffix);
    }

    std::string getTempFilename()
    {
        return getTempFilename(testing::TempDir());
    }

    std::string getGlobalTempDirectory()
    {
        if (std::getenv("TMP"))
        {
            return std::getenv("TMP");
        }
        else if (std::getenv("TEMP"))
        {
            return std::getenv("TEMP");
        }
        return "/tmp";
    }

    void removeTempFile(const std::string filename)
    {
        // in case of failure copy json file to some accessible directory
        if (testing::Test::HasFailure())
        {
            std::ifstream src(filename, std::ios::binary);
            std::string globalName = getTempFilename(getGlobalTempDirectory());
            std::ofstream dst(globalName, std::ios::binary);
            dst << src.rdbuf();
            dst.close();
            src.close();
            std::cerr << "Copied temporary file to " << globalName << std::endl;
        }
        std::remove(filename.c_str());
    }

    void captureStderr()
    {
        testing::internal::CaptureStderr();
    }

    std::string getCapturedStderr()
    {
        return testing::internal::GetCapturedStderr();
    }

    std::string joinPath(const std::string &p1, const std::string &p2)
    {
        return std::string(p1 + "/" + p2);
    }

    void createDir(const std::string &path) 
    {
        std::cerr << "TestUtils creating directory: " << path << std::endl;
        mkdir(path.c_str(), 0775);
    }
}
