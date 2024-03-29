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
#include "Params/BinFileParams.hpp"

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
    EXPECT_EXIT(binFileParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reading from input file: \\* Line 1, Column 23\n"
                              "  Missing ',' or '}' in object declaration\n");
}

TEST_F(BinFileTests, BinFileParamsReadInValidTestInvalidType)
{
    std::ofstream out(filename);
    out << "{\"filePath\":\"somepath\", \"binFileSizeInStructs\": \"invalidNumber\"}";
    out.close();
    std::cerr << filename << std::endl;

    BinFileParams binFileParamsRead;
    
    EXPECT_EXIT(binFileParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reached parsing arguments in BinFileParams: Value is not convertible to UInt64.\n");
}
