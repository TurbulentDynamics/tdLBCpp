//
//  GridTest.cpp
//  GridTest
//
//  Unit tests for GridParams
//

#include <cstdio>
#include <cstdlib>
#include <random>

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/GridParams.hpp"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"

class GridParamsTests : public ::testing::Test
{
protected:
    std::string filename;

public:
    GridParamsTests()
    {
        filename = TestUtils::getTempFilename();
    }
    ~GridParamsTests()
    {
        TestUtils::removeTempFile(filename);
    }
};

TEST_F(GridParamsTests, GridParamsWriteReadValidTest)
{
    GridParams gridParams = ParamsCommon::createGridParamsFixed();

    gridParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    GridParams gridParamsRead;
    gridParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(gridParams, gridParamsRead);
}

TEST_F(GridParamsTests, GridParamsWriteReadRandomValidTest)
{
    GridParams gridParams = ParamsCommon::createGridParamsRandom();

    gridParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    GridParams gridParamsRead;
    gridParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(gridParams, gridParamsRead);
}

TEST_F(GridParamsTests, GridParamsReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"ngx\":2,\"ngy\":3,\"x\":5,,,,,.....";
    out.close();
    std::cerr << filename << std::endl;

    GridParams gridParamsRead;
    EXPECT_EXIT(gridParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reading from input file: \\* Line 1, Column 24\n"
                              "  Missing '}' or object member name\n");
}

TEST_F(GridParamsTests, GridParamsReadInValidTestInvalidType)
{
    std::ofstream out(filename);
    out << "{\"ngx\":\"invalidInteger\"}";
    out.close();
    std::cerr << filename << std::endl;

    GridParams gridParamsRead;
    EXPECT_EXIT(gridParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reached parsing arguments in GridParams: Value is not convertible to UInt64.\n");
}
