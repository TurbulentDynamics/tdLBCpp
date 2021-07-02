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
#include "Params/Running.hpp"

#include "tdlbcpp/tests/utils.hpp"

class RunningParamsTests : public ::testing::Test
{
protected:
    std::string filename;

    void checkAllFields(RunningParams &expected, RunningParams &actual)
    {
        ASSERT_EQ(expected.step, actual.step) << "step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.num_steps, actual.num_steps) << "num_steps field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.angle, actual.angle) << "angle field has a wrong value after being written to a file and then read";
    }

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
    RunningParams runningParams;

    runningParams.step = 1;
    runningParams.num_steps = 2;
    runningParams.angle = 3;

    runningParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    RunningParams runningParamsRead;
    runningParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(runningParams, runningParamsRead);
}

TEST_F(RunningParamsTests, RunningParamsRandomWriteReadValidTest)
{
    RunningParams runningParams;

    runningParams.step = rand();
    runningParams.num_steps = rand();
    runningParams.angle = rand();

    runningParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    RunningParams runningParamsRead;
    runningParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(runningParams, runningParamsRead);
}

TEST_F(RunningParamsTests, RunningParamsReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"step\":2,\"num_steps\":3,\"angle\":5,,,,,.....";
    out.close();
    std::cerr << filename << std::endl;

    RunningParams runningParamsRead;
    testing::internal::CaptureStderr();
    runningParamsRead.getParamsFromJsonFile(filename);
    std::string capturedStdErr = testing::internal::GetCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: * Line 1, Column 35\n"
                              "  Missing '}' or object member name\n"
                              ", application will now exit\n")
        << "cerr should contain error";
}

TEST_F(RunningParamsTests, RunningParamsReadInValidTestInvalidType)
{
    std::ofstream out(filename);
    out << "{\"step\":\"invalidInteger\"}";
    out.close();
    std::cerr << filename << std::endl;

    RunningParams runningParamsRead;
    testing::internal::CaptureStderr();
    runningParamsRead.getParamsFromJsonFile(filename);
    std::string capturedStdErr = testing::internal::GetCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: "
                              "Value is not convertible to UInt64."
                              ", application will now exit\n")
        << "cerr should contain error";
}
