//
//  RunningTest.cpp
//  RunningTest
//
//  Unit tests for RunningParams
//

#include <cstdio>
#include <cstdlib>
#include <random>

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/RunningParams.hpp"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"

class RunningParamsTests : public ::testing::Test
{
protected:
    std::string filename;

public:
    RunningParamsTests()
    {
        filename = TestUtils::getTempFilename();
    }
    ~RunningParamsTests()
    {
        TestUtils::removeTempFile(filename);
    }
};

TEST_F(RunningParamsTests, RunningParamsWriteReadValidTest)
{
    RunningParams runningParams = ParamsCommon::createRunningParamsFixed();

    runningParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    RunningParams runningParamsRead;
    runningParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(runningParams, runningParamsRead);
}

TEST_F(RunningParamsTests, RunningParamsRandomWriteReadValidTest)
{
    RunningParams runningParams = ParamsCommon::createRunningParamsRandom();

    runningParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    RunningParams runningParamsRead;
    runningParamsRead.getParamsFromJsonFile(filename);

    ParamsCommon::checkAllFields(runningParams, runningParamsRead);
}

TEST_F(RunningParamsTests, RunningParamsReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"step\":2,\"num_steps\":3,\"angle\":5,,,,,.....";
    out.close();
    std::cerr << filename << std::endl;

    RunningParams runningParamsRead;
    EXPECT_EXIT(runningParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reading from input file: \\* Line 1, Column 35\n"
                              "  Missing '}' or object member name\n");
}

TEST_F(RunningParamsTests, RunningParamsReadInValidTestInvalidType)
{
    std::ofstream out(filename);
    out << "{\"step\":\"invalidInteger\"}";
    out.close();
    std::cerr << filename << std::endl;

    RunningParams runningParamsRead;
    EXPECT_EXIT(runningParamsRead.getParamsFromJsonFile(filename), testing::ExitedWithCode(1), 
    "Exception reached parsing arguments in RunningParams: Value is not convertible to UInt64.\n");;
}
