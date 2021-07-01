//
//  OutputParamsTest.cpp
//  OutputParamsTest
//
//  Unit tests for OutputParams
//

#include <cstdio>
#include <cstdlib>
#include <random>

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/OutputParams.hpp"

#include "tdlbcpp/tests/utils.hpp"

class OutputParamsTests : public ::testing::Test
{
protected:
    std::string filename;
    const int randomStringLength = 400;
    const int randomArraySize = 5;
    const double lower_bound = -10000;
    const double upper_bound = 10000;

    void checkAllFields(OrthoPlane &expected, OrthoPlane &actual)
    {
        ASSERT_EQ(expected.name_root, actual.name_root) << "name_root field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.repeat, actual.repeat) << "repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.cutAt, actual.cutAt) << "cutAt field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Q_output_len, actual.Q_output_len) << "Q_output_len field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.start_at_step, actual.start_at_step) << "start_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.end_at_repeat, actual.end_at_repeat) << "end_at_repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.use_half_float, actual.use_half_float) << "use_half_float field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
    }

    void checkAllFields(Volume &expected, Volume &actual)
    {
        ASSERT_EQ(expected.name_root, actual.name_root) << "name_root field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.repeat, actual.repeat) << "repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Q_output_len, actual.Q_output_len) << "Q_output_len field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.start_at_step, actual.start_at_step) << "start_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.end_at_repeat, actual.end_at_repeat) << "end_at_repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.use_half_float, actual.use_half_float) << "use_half_float field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
    }

    void checkAllFields(Angle &expected, Angle &actual)
    {
        ASSERT_EQ(expected.name_root, actual.name_root) << "name_root field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.repeat, actual.repeat) << "repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.degrees, actual.degrees) << "degrees field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Q_output_len, actual.Q_output_len) << "Q_output_len field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.start_at_step, actual.start_at_step) << "start_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.end_at_repeat, actual.end_at_repeat) << "end_at_repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.use_half_float, actual.use_half_float) << "use_half_float field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
    }

    void checkAllFields(PlaneAtAngle &expected, PlaneAtAngle &actual)
    {
        ASSERT_EQ(expected.name_root, actual.name_root) << "name_root field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.degrees, actual.degrees) << "degrees field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.tolerance, actual.tolerance) << "tolerance field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.cutAt, actual.cutAt) << "cutAt field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Q_output_len, actual.Q_output_len) << "Q_output_len field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.start_at_step, actual.start_at_step) << "start_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.end_at_repeat, actual.end_at_repeat) << "end_at_repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.use_half_float, actual.use_half_float) << "use_half_float field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
    }

    void checkAllFields(Sector &expected, Sector &actual)
    {
        ASSERT_EQ(expected.name_root, actual.name_root) << "name_root field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.repeat, actual.repeat) << "repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.angle_infront_blade, actual.angle_infront_blade) << "angle_infront_blade field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.angle_behind_blade, actual.angle_behind_blade) << "angle_behind_blade field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Q_output_len, actual.Q_output_len) << "Q_output_len field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.start_at_step, actual.start_at_step) << "start_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.end_at_repeat, actual.end_at_repeat) << "end_at_repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.use_half_float, actual.use_half_float) << "use_half_float field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
    }

    void checkAllFields(OutputParams &expected, OutputParams &actual)
    {
        ASSERT_EQ(expected.rootDir, actual.rootDir) << "rootDir field has a wrong value after being written to a file and then read";

        ASSERT_EQ(expected.XY_planes.size(), actual.XY_planes.size()) << "XY_planes field has wrong size after being written to a file and then read";
        for (size_t i = 0; i < expected.XY_planes.size(); i++)
        {
            checkAllFields(expected.XY_planes[i], actual.XY_planes[i]);
        }

        ASSERT_EQ(expected.XZ_planes.size(), actual.XZ_planes.size()) << "XZ_planes field has wrong size after being written to a file and then read";
        for (size_t i = 0; i < expected.XZ_planes.size(); i++)
        {
            checkAllFields(expected.XZ_planes[i], actual.XZ_planes[i]);
        }

        ASSERT_EQ(expected.YZ_planes.size(), actual.YZ_planes.size()) << "YZ_planes field has wrong size after being written to a file and then read";
        for (size_t i = 0; i < expected.YZ_planes.size(); i++)
        {
            checkAllFields(expected.YZ_planes[i], actual.YZ_planes[i]);
        }

        ASSERT_EQ(expected.capture_at_blade_angle.size(), actual.capture_at_blade_angle.size()) << "capture_at_blade_angle field has wrong size after being written to a file and then read";
        for (size_t i = 0; i < expected.capture_at_blade_angle.size(); i++)
        {
            checkAllFields(expected.capture_at_blade_angle[i], actual.capture_at_blade_angle[i]);
        }

        ASSERT_EQ(expected.YZ_plane_when_angle.size(), actual.YZ_plane_when_angle.size()) << "YZ_plane_when_angle field has wrong size after being written to a file and then read";
        for (size_t i = 0; i < expected.YZ_plane_when_angle.size(); i++)
        {
            checkAllFields(expected.YZ_plane_when_angle[i], actual.YZ_plane_when_angle[i]);
        }

        ASSERT_EQ(expected.volumes.size(), actual.volumes.size()) << "volumes field has wrong size after being written to a file and then read";
        for (size_t i = 0; i < expected.volumes.size(); i++)
        {
            checkAllFields(expected.volumes[i], actual.volumes[i]);
        }
    }

public:
    OutputParamsTests()
    {
        filename = TestUtils::getTempFilename("_to_delete.json");
    }
    ~OutputParamsTests()
    {
        TestUtils::removeTempFile(filename);
    }
};

TEST_F(OutputParamsTests, OutputParamsWriteReadValidTest)
{
    OutputParams outputParams("test1");
    outputParams.add_XY_plane("dirTest1", 1, 2, 3, 4, 5, true);
    outputParams.add_XY_plane("dirTest2", 6, 7, 8, 9, 10, false);

    outputParams.add_XZ_plane("dirTest3", 11, 12, 13, 14, 15, true);
    outputParams.add_XZ_plane("dirTest4", 16, 17, 18, 19, 20, false);
    outputParams.add_XZ_plane("dirTest5", 21, 22, 23, 24, 25, true);

    outputParams.add_YZ_plane("dirTest6", 31, 32, 33, 34, 35, true);
    outputParams.add_YZ_plane("dirTest7", 36, 37, 38, 39, 40, false);
    outputParams.add_YZ_plane("dirTest8", 41, 42, 43, 44, 45, true);
    outputParams.add_YZ_plane("dirTest9", 46, 47, 48, 49, 50, false);

    outputParams.add_angle("dirTest10", 51, 52.0, 53, 54, 55, true);
    outputParams.add_angle("dirTest11", 56, 52.0, 53, 54, 55, false);
    outputParams.add_angle("dirTest12", 61, 62.0, 63, 64, 65, true);
    outputParams.add_angle("dirTest13", 66, 67.0, 68, 69, 70, false);
    outputParams.add_angle("dirTest14", 71, 72.0, 73, 74, 75, true);

    outputParams.add_YZ_plane_at_angle("dirTest15", 76.0, 77.0, 78, 79, 80, 81, false);
    outputParams.add_YZ_plane_at_angle("dirTest16", 82.0, 83.0, 84, 85, 86, 87, true);
    outputParams.add_YZ_plane_at_angle("dirTest17", 88.0, 89.0, 90, 91, 92, 93, false);
    outputParams.add_YZ_plane_at_angle("dirTest18", 94.0, 95.0, 96, 97, 98, 99, true);
    outputParams.add_YZ_plane_at_angle("dirTest19", 100.0, 101.0, 102, 103, 104, 105, false);
    outputParams.add_YZ_plane_at_angle("dirTest20", 106.0, 107.0, 108, 109, 110, 111, true);

    outputParams.add_volume("dirTest21", 112, 113, 114, 115, false);

    outputParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    OutputParams outputParamsRead("test2");
    outputParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(outputParams, outputParamsRead);
}

TEST_F(OutputParamsTests, OutputParamsRandomWriteReadValidTest)
{
    OutputParams outputParams(TestUtils::random_string(randomStringLength));
    for (int i = 0; i < rand() % randomArraySize; i++)
    {
        outputParams.add_XY_plane(TestUtils::random_string(randomStringLength), rand(), rand(), rand(), rand(), rand(), ((rand() & 1) == 1));
    }
    for (int i = 0; i < rand() % randomArraySize; i++)
    {
        outputParams.add_XZ_plane(TestUtils::random_string(randomStringLength), rand(), rand(), rand(), rand(), rand(), ((rand() & 1) == 1));
    }
    for (int i = 0; i < rand() % randomArraySize; i++)
    {
        outputParams.add_YZ_plane(TestUtils::random_string(randomStringLength), rand(), rand(), rand(), rand(), rand(), ((rand() & 1) == 1));
    }

    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    for (int i = 0; i < rand() % randomArraySize; i++)
    {
        outputParams.add_angle(TestUtils::random_string(randomStringLength), rand(), unif(re), rand(), rand(), rand(), ((rand() & 1) == 1));
    }

    for (int i = 0; i < rand() % randomArraySize; i++)
    {
        outputParams.add_YZ_plane_at_angle(TestUtils::random_string(randomStringLength), unif(re), unif(re), rand(), rand(), rand(), rand(), ((rand() & 1) == 1));
    }

    for (int i = 0; i < rand() % randomArraySize; i++)
    {
        outputParams.add_volume(TestUtils::random_string(randomStringLength), rand(), rand(), rand(), rand(), ((rand() & 1) == 1));
    }

    outputParams.writeParamsToJsonFile(filename);
    std::cerr << filename << std::endl;

    OutputParams outputParamsRead("test2");
    outputParamsRead.getParamsFromJsonFile(filename);

    checkAllFields(outputParams, outputParamsRead);
}

TEST_F(OutputParamsTests, OutputParamsInvalidTest)
{
    std::ofstream out(filename);
    out << "{\"XY_plane\":[}";
    out.close();
    std::cerr << filename << std::endl;

    OutputParams outputParamsRead("test");
    testing::internal::CaptureStderr();
    outputParamsRead.getParamsFromJsonFile(filename);
    std::string capturedStdErr = testing::internal::GetCapturedStderr();

    ASSERT_EQ(capturedStdErr, "Unhandled Exception reached parsing arguments: * Line 1, Column 14\n"
                              "  Syntax error: value, object or array expected.\n"
                              ", application will now exit\n")
        << "cerr should contain error";
}