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

class ComputeUnitParamsTests : public ::testing::Test
{
protected:
    std::string filename;
    const int randomStringLength = 400;

    void checkAllFields(ComputeUnitParams &expected, ComputeUnitParams &actual)
    {
        ASSERT_EQ(expected.idi, actual.idi) << "idi field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.idj, actual.idj) << "idj field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.idk, actual.idk) << "idk field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.x, actual.x) << "x field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.y, actual.y) << "y field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.z, actual.z) << "z field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.i0, actual.i0) << "i0 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.j0, actual.j0) << "j0 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.k0, actual.k0) << "k0 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.ghost, actual.ghost) << "ghost field has a wrong value after being written to a file and then read";
    }

public:
    ComputeUnitParamsTests()
    {
        filename = TestUtils::getTempFilename("_to_delete.json");
    }
    ~ComputeUnitParamsTests()
    {
        TestUtils::removeTempFile(filename);
    }
};

TEST_F(ComputeUnitParamsTests, ComputeUnitParamsWriteReadValidTest)
{
    ComputeUnitParams computeUnitParams;
    computeUnitParams.idi = 1;
    computeUnitParams.idj = 2;
    computeUnitParams.idk = 3;
    computeUnitParams.x = 4;
    computeUnitParams.y = 5;
    computeUnitParams.z = 6;
    computeUnitParams.i0 = 7;
    computeUnitParams.j0 = 8;
    computeUnitParams.k0 = 9;
    computeUnitParams.ghost = 10;

    computeUnitParams.writeParams(filename);
    std::cerr << filename << std::endl;

    ComputeUnitParams computeUnitParamsRead;
    computeUnitParamsRead.getParamFromJson(filename);

    checkAllFields(computeUnitParams, computeUnitParamsRead);
}

TEST_F(ComputeUnitParamsTests, ComputeUnitParamsWriteReadValidRandomTest)
{
    ComputeUnitParams computeUnitParams;
    computeUnitParams.idi = rand();
    computeUnitParams.idj = rand();
    computeUnitParams.idk = rand();
    computeUnitParams.x = rand();
    computeUnitParams.y = rand();
    computeUnitParams.z = rand();
    computeUnitParams.i0 = rand();
    computeUnitParams.j0 = rand();
    computeUnitParams.k0 = rand();
    computeUnitParams.ghost = rand();

    computeUnitParams.writeParams(filename);
    std::cerr << filename << std::endl;

    ComputeUnitParams computeUnitParamsRead;
    computeUnitParamsRead.getParamFromJson(filename);

    checkAllFields(computeUnitParams, computeUnitParamsRead);
}

TEST_F(ComputeUnitParamsTests, ComputeUnitParamsReadInValidTest)
{
    std::ofstream out(filename);
    out << "{\"idi\":2";
    out.close();
    std::cerr << filename << std::endl;

    ComputeUnitParams computeUnitParamsRead;
    testing::internal::CaptureStderr();
    computeUnitParamsRead.getParamFromJson(filename);
    std::string capturedStdErr = testing::internal::GetCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: * Line 1, Column 9\n"
                              "  Missing ',' or '}' in object declaration\n"
                              ", application will now exit\n")
        << "cerr should contain error";
}