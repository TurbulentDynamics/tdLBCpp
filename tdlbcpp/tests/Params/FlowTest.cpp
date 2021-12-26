//
//  FlowTest.cpp
//  FlowTest
//
//  Unit tests for FlowParams<typename T>
//

#include <cstdio>
#include <cstdlib>

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/FlowParams.hpp"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"

class FlowParamsTests : public ::testing::Test
{
protected:
    std::string filename;

public:
    FlowParamsTests()
    {
        filename = TestUtils::getTempFilename();
    }
    ~FlowParamsTests()
    {
        TestUtils::removeTempFile(filename);
    }
};

TEST_F(FlowParamsTests, FlowDoubleWriteReadValidTest)
{
    FlowParams<double> flowParams = ParamsCommon::createFlowParamsFixed();

    flowParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    FlowParams<double> flowParamsRead;
    flowParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(flowParams, flowParamsRead);
}

TEST_F(FlowParamsTests, FlowDoubleWriteReadRandomValidTest)
{
    FlowParams<double> flowParams = ParamsCommon::createFlowParamsWithRandomValues<double>();

    flowParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    FlowParams<double> flowParamsRead;
    flowParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(flowParams, flowParamsRead);
}

TEST_F(FlowParamsTests, FlowFloatWriteReadRandomValidTest)
{
    FlowParams<float> flowParams = ParamsCommon::createFlowParamsWithRandomValues<float>();

    flowParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    FlowParams<float> flowParamsRead;
    flowParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(flowParams, flowParamsRead);
}

TEST_F(FlowParamsTests, FlowParamsReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"initialRho\":2";
    out.close();
    std::cerr << filename << std::endl;

    FlowParams<double> flowParamsRead;
    EXPECT_EXIT(flowParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reading from input file: \\* Line 1, Column 16\n"
                              "  Missing ',' or '}' in object declaration\n");
}

TEST_F(FlowParamsTests, FlowParamsReadInValidTestInvalidType)
{
    std::ofstream out(filename);
    out << "{\"initialRho\":\"test\"}";
    out.close();
    std::cerr << filename << std::endl;

    FlowParams<double> flowParamsRead;
    EXPECT_EXIT(flowParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reached parsing arguments in FlowParams: Value is not convertible to double.\n");
}
