//
//  ComputeUnitParamsTest.cpp
//  ComputeUnitParamsTest
//
//  Unit tests for ComputeUnitParams
//

#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/ComputeUnitParams.hpp"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"

class ComputeUnitParamsTests : public ::testing::Test
{
protected:
    std::string filename;

public:
    ComputeUnitParamsTests()
    {
        filename = TestUtils::getTempFilename();
    }
    ~ComputeUnitParamsTests()
    {
        TestUtils::removeTempFile(filename);
    }
};

TEST_F(ComputeUnitParamsTests, ComputeUnitParamsWriteReadValidTest)
{
    ComputeUnitParams computeUnitParams = ParamsCommon::createComputeUnitParamsFixed();

    computeUnitParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    ComputeUnitParams computeUnitParamsRead;
    computeUnitParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(computeUnitParams, computeUnitParamsRead);
}

TEST_F(ComputeUnitParamsTests, ComputeUnitParamsWriteReadValidRandomTest)
{
    ComputeUnitParams computeUnitParams = ParamsCommon::createComputeUnitParamsRandom();

    computeUnitParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    ComputeUnitParams computeUnitParamsRead;
    computeUnitParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(computeUnitParams, computeUnitParamsRead);
}

TEST_F(ComputeUnitParamsTests, ComputeUnitParamsReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"idi\":2";
    out.close();
    std::cerr << filename << std::endl;

    ComputeUnitParams computeUnitParamsRead;
    EXPECT_EXIT(computeUnitParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reading from input file: \\* Line 1, Column 9\n"
                              "  Missing ',' or '}' in object declaration\n");
}

TEST_F(ComputeUnitParamsTests, ComputeUnitParamsReadInValidTestInvalidType)
{
    std::ofstream out(filename);
    out << "{\"idi\":\"invalidNumber\"}";
    out.close();
    std::cerr << filename << std::endl;

    ComputeUnitParams computeUnitParamsRead;
    EXPECT_EXIT(computeUnitParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reached parsing arguments in ComputeUnitParams: Value is not convertible to Int.\n");
}
