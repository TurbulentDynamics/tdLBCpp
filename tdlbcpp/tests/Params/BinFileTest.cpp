//
//  BinFileTest.cpp
//  BinFileTest
//
//
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
    const int randomStringLength = 400;

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
        const testing::TestInfo *const test_info =
            testing::UnitTest::GetInstance()->current_test_info();
        std::string testName = test_info->name();
        filename = TestUtils::getTempFilename(testName + "_to_delete.json");
    }
    ~BinFileTests()
    {
        TestUtils::removeTempFile(filename);
    }
};

TEST_F(BinFileTests, BinFileWriteReadValidTest)
{
    BinFileParams binFileFormat;
    binFileFormat.filePath = "test";
    binFileFormat.name = "test1";
    binFileFormat.note = "test2";
    binFileFormat.structName = "test3";
    binFileFormat.binFileSizeInStructs = 2;
    binFileFormat.coordsType = "test4";
    binFileFormat.hasGridtCoords = true;
    binFileFormat.hasColRowtCoords = true;
    binFileFormat.QDataType = "test5";
    binFileFormat.QOutputLength = 0x8fffffff;

    binFileFormat.writeParams(filename);
    std::cerr << filename << std::endl;

    BinFileParams binFileFormatRead;
    binFileFormatRead.getParamFromJson(filename);

    checkAllFields(binFileFormatRead, binFileFormat);
}

TEST_F(BinFileTests, BinFileWriteReadValidRandomTest)
{
    BinFileParams binFileFormat;
    binFileFormat.filePath = TestUtils::random_string(randomStringLength);
    binFileFormat.name = TestUtils::random_string(randomStringLength);
    binFileFormat.note = TestUtils::random_string(randomStringLength);
    binFileFormat.structName = TestUtils::random_string(randomStringLength);
    binFileFormat.binFileSizeInStructs = rand();
    binFileFormat.coordsType = TestUtils::random_string(randomStringLength);
    binFileFormat.hasGridtCoords = (rand() & 1) == 1;
    binFileFormat.hasColRowtCoords = (rand() & 1) == 1;
    binFileFormat.QDataType = TestUtils::random_string(randomStringLength);
    binFileFormat.QOutputLength = rand();

    binFileFormat.writeParams(filename);
    std::cerr << filename << std::endl;

    BinFileParams binFileFormatRead;
    binFileFormatRead.getParamFromJson(filename);

    checkAllFields(binFileFormatRead, binFileFormat);
}

TEST_F(BinFileTests, BinFileReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"filePath\":\"somepath\"}";
    out.close();
    std::cerr << filename << std::endl;

    BinFileParams binFileFormatRead;
    binFileFormatRead.getParamFromJson(filename);

    ASSERT_EQ(binFileFormatRead.filePath, "somepath") << "filePath field has a wrong value";
}