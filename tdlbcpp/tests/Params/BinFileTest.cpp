//
//  BinFileTest.cpp
//  BinFileTest
//
//  Unit tests for BinFileParams
//

#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/BinFile.hpp"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"

class BinFileTests : public ::testing::Test
{
protected:
    std::string filename;

public:
    BinFileTests()
    {
        filename = TestUtils::getTempFilename();
    }
    ~BinFileTests()
    {
        TestUtils::removeTempFile(filename);
    }
};

TEST_F(BinFileTests, BinFileWriteReadValidTest)
{
    BinFileParams binFileParams = ParamsCommon::createBinFileParamsFixed();

    binFileParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    BinFileParams binFileParamsRead;
    binFileParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(binFileParamsRead, binFileParams);
}

TEST_F(BinFileTests, BinFileWriteReadValidRandomTest)
{
    BinFileParams binFileParams = ParamsCommon::createBinFileParamsRandom();

    binFileParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    BinFileParams binFileParamsRead;
    binFileParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(binFileParamsRead, binFileParams);
}

TEST_F(BinFileTests, BinFileParamsReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"filePath\":\"somepath\"";
    out.close();
    std::cerr << filename << std::endl;

    BinFileParams binFileParamsRead;
    TestUtils::captureStderr();
    binFileParamsRead.getParamsFromJsonFile(filename);
    std::string capturedStdErr = TestUtils::getCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: * Line 1, Column 23\n"
                              "  Missing ',' or '}' in object declaration\n"
                              ", application will now exit\n")
        << "cerr should contain error";
}

TEST_F(BinFileTests, BinFileParamsReadInValidTestInvalidType)
{
    std::ofstream out(filename);
    out << "{\"filePath\":\"somepath\", \"binFileSizeInStructs\": \"invalidNumber\"}";
    out.close();
    std::cerr << filename << std::endl;

    BinFileParams binFileParamsRead;
    TestUtils::captureStderr();
    binFileParamsRead.getParamsFromJsonFile(filename);
    std::string capturedStdErr = TestUtils::getCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: "
                              "Value is not convertible to UInt64."
                              ", application will now exit\n")
        << "cerr should contain error";
}
