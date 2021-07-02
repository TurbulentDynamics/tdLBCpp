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
    TestUtils::captureStderr();
    computeUnitParamsRead.getParamsFromJsonFile(filename);
    std::string capturedStdErr = TestUtils::getCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: * Line 1, Column 9\n"
                              "  Missing ',' or '}' in object declaration\n"
                              ", application will now exit\n")
        << "cerr should contain error";
}

TEST_F(ComputeUnitParamsTests, ComputeUnitParamsReadInValidTestInvalidType)
{
    std::ofstream out(filename);
    out << "{\"idi\":\"invalidNumber\"}";
    out.close();
    std::cerr << filename << std::endl;

    ComputeUnitParams computeUnitParamsRead;
    TestUtils::captureStderr();
    computeUnitParamsRead.getParamsFromJsonFile(filename);
    std::string capturedStdErr = TestUtils::getCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: "
                              "Value is not convertible to Int."
                              ", application will now exit\n")
        << "cerr should contain error";
}
