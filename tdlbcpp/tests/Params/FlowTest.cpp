//
//  FlowTest.cpp
//  FlowTest
//
//  Unit tests for FlowParams<typename T>
//

#include <cstdio>
#include <cstdlib>
#include <random>

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/Flow.hpp"

#include "tdlbcpp/tests/utils.hpp"

class FlowParamsTests : public ::testing::Test
{
protected:
    std::string filename;

    template <typename T>
    FlowParams<T> createFlowParamsWithRandomValues()
    {
        FlowParams<T> flowParams;

        std::uniform_real_distribution<double> unif(TestUtils::randomDoubleLowerBound, TestUtils::randomDoubleUpperBound);
        std::default_random_engine re;

        flowParams.initialRho = (T)unif(re);
        flowParams.reMNonDimensional = (T)unif(re);
        flowParams.uav = (T)unif(re);
        flowParams.cs0 = (T)unif(re);
        flowParams.g3 = (T)unif(re);
        flowParams.nu = (T)unif(re);
        flowParams.fx0 = (T)unif(re);
        flowParams.Re_m = (T)unif(re);
        flowParams.Re_f = (T)unif(re);
        flowParams.uf = (T)unif(re);
        flowParams.alpha = (T)unif(re);
        flowParams.beta = (T)unif(re);
        flowParams.useLES = (rand() & 1) == 1;
        flowParams.collision = TestUtils::random_string(TestUtils::randomStringLength);
        flowParams.streaming = TestUtils::random_string(TestUtils::randomStringLength);

        return flowParams;
    }

    template <typename T>
    void checkAllFields(FlowParams<T> &expected, FlowParams<T> &actual)
    {
        ASSERT_EQ(expected.initialRho, actual.initialRho) << "initialRho field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.reMNonDimensional, actual.reMNonDimensional) << "reMNonDimensional field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.uav, actual.uav) << "uav field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.cs0, actual.cs0) << "cs0 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.g3, actual.g3) << "g3 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.nu, actual.nu) << "nu field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.fx0, actual.fx0) << "fx0 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Re_m, actual.Re_m) << "Re_m field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Re_f, actual.Re_f) << "Re_f field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.uf, actual.uf) << "uf field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.alpha, actual.alpha) << "alpha field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.beta, actual.beta) << "beta field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.useLES, actual.useLES) << "useLES field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.collision, actual.collision) << "collision field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.streaming, actual.streaming) << "streaming field has a wrong value after being written to a file and then read";
    }

public:
    FlowParamsTests()
    {
        filename = TestUtils::getTempFilename("_to_delete.json");
    }
    ~FlowParamsTests()
    {
        TestUtils::removeTempFile(filename);
    }
};

TEST_F(FlowParamsTests, FlowDoubleWriteReadValidTest)
{
    FlowParams<double> flowParams;

    flowParams.initialRho = 0.0;
    flowParams.reMNonDimensional = 0.1;
    flowParams.uav = 0.2;
    flowParams.cs0 = 0.3;
    flowParams.g3 = 0.4;
    flowParams.nu = 0.5;
    flowParams.fx0 = 0.6;
    flowParams.Re_m = 0.7;
    flowParams.Re_f = 0.8;
    flowParams.uf = 0.9;
    flowParams.alpha = 1.0;
    flowParams.beta = 1.1;
    flowParams.useLES = true;
    flowParams.collision = "test1";
    flowParams.streaming = "test2";

    flowParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    FlowParams<double> flowParamsRead;
    flowParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(flowParams, flowParamsRead);
}

TEST_F(FlowParamsTests, FlowDoubleWriteReadRandomValidTest)
{
    FlowParams<double> flowParams = createFlowParamsWithRandomValues<double>();

    flowParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    FlowParams<double> flowParamsRead;
    flowParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(flowParams, flowParamsRead);
}

TEST_F(FlowParamsTests, FlowFloatWriteReadRandomValidTest)
{
    FlowParams<float> flowParams = createFlowParamsWithRandomValues<float>();

    flowParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    FlowParams<float> flowParamsRead;
    flowParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(flowParams, flowParamsRead);
}

TEST_F(FlowParamsTests, FlowParamsReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"initialRho\":2";
    out.close();
    std::cerr << filename << std::endl;

    FlowParams<double> flowParamsRead;
    testing::internal::CaptureStderr();
    flowParamsRead.getParamsFromJsonFile(filename);
    std::string capturedStdErr = testing::internal::GetCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: * Line 1, Column 16\n"
                              "  Missing ',' or '}' in object declaration\n"
                              ", application will now exit\n")
        << "cerr should contain error";
}

TEST_F(FlowParamsTests, FlowParamsReadInValidTestInvalidType)
{
    std::ofstream out(filename);
    out << "{\"initialRho\":\"test\"}";
    out.close();
    std::cerr << filename << std::endl;

    FlowParams<double> flowParamsRead;
    testing::internal::CaptureStderr();
    flowParamsRead.getParamsFromJsonFile(filename);
    std::string capturedStdErr = testing::internal::GetCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: "
                              "Value is not convertible to double."
                              ", application will now exit\n")
        << "cerr should contain error";
}
