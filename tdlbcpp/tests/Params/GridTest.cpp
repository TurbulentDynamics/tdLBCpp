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
#include "Params/Grid.hpp"

#include "tdlbcpp/tests/utils.hpp"

class GridParamsTests : public ::testing::Test
{
protected:
    std::string filename;

    void checkAllFields(GridParams &expected, GridParams &actual)
    {
        ASSERT_EQ(expected.ngx, actual.ngx) << "ngx field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.ngy, actual.ngy) << "ngy field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.ngz, actual.ngz) << "ngz field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.x, actual.x) << "x field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.y, actual.y) << "y field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.z, actual.z) << "z field has a wrong value after being written to a file and then read";
    }

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
    GridParams gridParams;

    gridParams.ngx = 1;
    gridParams.ngy = 2;
    gridParams.ngz = 3;
    gridParams.x = 4;
    gridParams.y = 5;
    gridParams.z = 6;

    gridParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    GridParams gridParamsRead;
    gridParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(gridParams, gridParamsRead);
}

TEST_F(GridParamsTests, GridParamsReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"ngx\":2,\"ngy\":3,\"x\":5,,,,,.....";
    out.close();
    std::cerr << filename << std::endl;

    GridParams gridParamsRead;
    TestUtils::captureStderr();
    gridParamsRead.getParamsFromJsonFile(filename);
    std::string capturedStdErr = TestUtils::getCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: * Line 1, Column 24\n"
                              "  Missing '}' or object member name\n"
                              ", application will now exit\n")
        << "cerr should contain error";
}

TEST_F(GridParamsTests, GridParamsReadInValidTestInvalidType)
{
    std::ofstream out(filename);
    out << "{\"ngx\":\"invalidInteger\"}";
    out.close();
    std::cerr << filename << std::endl;

    GridParams gridParamsRead;
    TestUtils::captureStderr();
    gridParamsRead.getParamsFromJsonFile(filename);
    std::string capturedStdErr = TestUtils::getCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: "
                              "Value is not convertible to Int."
                              ", application will now exit\n")
        << "cerr should contain error";
}
