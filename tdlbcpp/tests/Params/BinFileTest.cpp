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

class BinFileTests : public ::testing::Test
{
protected:
    std::string filename;

    void checkAllFields(BinFileParams &expected, BinFileParams &actual)
    {
        ASSERT_EQ(expected.filePath, actual.filePath) << "filePath field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.name, actual.name) << "name field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.note, actual.note) << "note field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.structName, actual.structName) << "structName field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.binFileSizeInStructs, actual.binFileSizeInStructs) << "binFileSizeInStructs field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.coordsType, actual.coordsType) << "coordsType field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.hasGridtCoords, actual.hasGridtCoords) << "hasGridtCoords field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.hasColRowtCoords, actual.hasColRowtCoords) << "hasColRowtCoords field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QOutputLength, actual.QOutputLength) << "QOutputLength field has a wrong value after being written to a file and then read";
    }

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
    BinFileParams binFileParams;
    binFileParams.filePath = "test";
    binFileParams.name = "test1";
    binFileParams.note = "test2";
    binFileParams.structName = "test3";
    binFileParams.binFileSizeInStructs = 2;
    binFileParams.coordsType = "test4";
    binFileParams.hasGridtCoords = true;
    binFileParams.hasColRowtCoords = true;
    binFileParams.QDataType = "test5";
    binFileParams.QOutputLength = 0x8fffffff;

    binFileParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    BinFileParams binFileParamsRead;
    binFileParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(binFileParamsRead, binFileParams);
}

TEST_F(BinFileTests, BinFileWriteReadValidRandomTest)
{
    BinFileParams binFileParams;
    binFileParams.filePath = TestUtils::random_string(TestUtils::randomStringLength);
    binFileParams.name = TestUtils::random_string(TestUtils::randomStringLength);
    binFileParams.note = TestUtils::random_string(TestUtils::randomStringLength);
    binFileParams.structName = TestUtils::random_string(TestUtils::randomStringLength);
    binFileParams.binFileSizeInStructs = rand();
    binFileParams.coordsType = TestUtils::random_string(TestUtils::randomStringLength);
    binFileParams.hasGridtCoords = (rand() & 1) == 1;
    binFileParams.hasColRowtCoords = (rand() & 1) == 1;
    binFileParams.QDataType = TestUtils::random_string(TestUtils::randomStringLength);
    binFileParams.QOutputLength = rand();

    binFileParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    BinFileParams binFileParamsRead;
    binFileParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(binFileParamsRead, binFileParams);
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
