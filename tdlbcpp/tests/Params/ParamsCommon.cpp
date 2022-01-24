#include "gtest/gtest.h"

#if WITH_MPI == 1
#include <mpi.h>
#endif

#include "Header.h"
#include "ParamsCommon.hpp"
#include "Params/CheckpointParams.hpp"
#include "Params/BinFileParams.hpp"
#include "Params/ComputeUnitParams.hpp"
#include "Params/FlowParams.hpp"
#include "Params/GridParams.hpp"
#include "Params/RunningParams.hpp"
#include "Params/OutputParams.hpp"

#include "tdlbcpp/tests/utils.hpp"

namespace ParamsCommon
{
    // CheckpointParams helper methods
    CheckpointParams createCheckpointParamsFixed()
    {
        CheckpointParams checkpointParams;
        checkpointParams.startWithCheckpoint = true;
        checkpointParams.checkpointLoadFromDir = "test1";
        checkpointParams.checkpointRepeat = 1;
        checkpointParams.checkpointWriteRootDir = "test2";
        checkpointParams.checkpointWriteDirPrefix = "test3";
        checkpointParams.checkpointWriteDirAppendTime = true;
        return checkpointParams;
    }

    CheckpointParams createCheckpointParamsRandom()
    {
        CheckpointParams checkpointParams;
        checkpointParams.startWithCheckpoint = (rand() & 1) == 1;
        checkpointParams.checkpointLoadFromDir = TestUtils::random_string(TestUtils::randomStringLength);
        checkpointParams.checkpointRepeat = rand();
        checkpointParams.checkpointWriteRootDir = TestUtils::random_string(TestUtils::randomStringLength);
        checkpointParams.checkpointWriteDirPrefix = TestUtils::random_string(TestUtils::randomStringLength);
        checkpointParams.checkpointWriteDirAppendTime = (rand() & 1) == 1;
        return checkpointParams;
    }

    void checkAllFields(CheckpointParams &expected, CheckpointParams &actual)
    {
        ASSERT_EQ(expected.startWithCheckpoint, actual.startWithCheckpoint) << "startWithCheckpoint field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.checkpointLoadFromDir, actual.checkpointLoadFromDir) << "checkpointLoadFromDir field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.checkpointRepeat, actual.checkpointRepeat) << "checkpointRepeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.checkpointWriteRootDir, actual.checkpointWriteRootDir) << "checkpointWriteRootDir field has a wrong value after being written to a file and then read";
    }

    // BinFileParams helper methods
    BinFileParams createBinFileParamsFixed()
    {
        BinFileParams binFileParams;
        binFileParams.filePath = "test";
        binFileParams.name = "test1";
        binFileParams.note = "test2";
        binFileParams.structName = "test3";
        binFileParams.binFileSizeInStructs = 2;
        binFileParams.coordsType = "test4";
        binFileParams.hasGridtCoords = true;
        binFileParams.hasColRowtCoords = true;
        binFileParams.QDataType = "test5";
        binFileParams.QOutputLength = 0x8fffffff;
        return binFileParams;
    }

    BinFileParams createBinFileParamsRandom()
    {
        BinFileParams binFileParams;
        binFileParams.filePath = TestUtils::random_string(TestUtils::randomStringLength);
        binFileParams.name = TestUtils::random_string(TestUtils::randomStringLength);
        binFileParams.note = TestUtils::random_string(TestUtils::randomStringLength);
        binFileParams.structName = TestUtils::random_string(TestUtils::randomStringLength);
        binFileParams.binFileSizeInStructs = rand();
        binFileParams.coordsType = TestUtils::random_string(TestUtils::randomStringLength);
        binFileParams.hasGridtCoords = (rand() & 1) == 1;
        binFileParams.hasColRowtCoords = (rand() & 1) == 1;
        binFileParams.QDataType = TestUtils::random_string(TestUtils::randomStringLength);
        binFileParams.QOutputLength = rand();
        return binFileParams;
    }

    void checkAllFields(BinFileParams &expected, BinFileParams &actual)
    {
        ASSERT_EQ(expected.filePath, actual.filePath) << "filePath field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.name, actual.name) << "name field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.note, actual.note) << "note field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.structName, actual.structName) << "structName field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.binFileSizeInStructs, actual.binFileSizeInStructs) << "binFileSizeInStructs field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.coordsType, actual.coordsType) << "coordsType field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.hasGridtCoords, actual.hasGridtCoords) << "hasGridtCoords field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.hasColRowtCoords, actual.hasColRowtCoords) << "hasColRowtCoords field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QOutputLength, actual.QOutputLength) << "QOutputLength field has a wrong value after being written to a file and then read";
    }

    // ComputeUnitParams helper methods
    ComputeUnitParams createComputeUnitParamsFixed()
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
        return computeUnitParams;
    }

    ComputeUnitParams createComputeUnitParamsRandom()
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
        return computeUnitParams;
    }

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

    // FlowParams helper methods
    FlowParams<double> createFlowParamsFixed()
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
        return flowParams;
    }

    // GridParams helper methods
    GridParams createGridParamsFixed()
    {
        GridParams gridParams;

        gridParams.ngx = 1;
        gridParams.ngy = 2;
        gridParams.ngz = 3;
        gridParams.x = 4;
        gridParams.y = 5;
        gridParams.z = 6;

        return gridParams;
    }

    GridParams createGridParamsRandom()
    {
        GridParams gridParams;

        gridParams.ngx = rand();
        gridParams.ngy = rand();
        gridParams.ngz = rand();
        gridParams.x = rand();
        gridParams.y = rand();
        gridParams.z = rand();

        return gridParams;
    }

    void checkAllFields(GridParams &expected, GridParams &actual)
    {
        ASSERT_EQ(expected.ngx, actual.ngx) << "ngx field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.ngy, actual.ngy) << "ngy field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.ngz, actual.ngz) << "ngz field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.x, actual.x) << "x field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.y, actual.y) << "y field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.z, actual.z) << "z field has a wrong value after being written to a file and then read";
    }

    // RunningParams helper functions
    RunningParams createRunningParamsFixed()
    {
        RunningParams runningParams;

        runningParams.step = 1;
        runningParams.num_steps = 2;
        runningParams.angle = 3;

        return runningParams;
    }

    RunningParams createRunningParamsRandom()
    {
        RunningParams runningParams;

        runningParams.step = rand();
        runningParams.num_steps = rand();
        runningParams.angle = rand();

        return runningParams;
    }

    void checkAllFields(RunningParams &expected, RunningParams &actual)
    {
        ASSERT_EQ(expected.step, actual.step) << "step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.num_steps, actual.num_steps) << "num_steps field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.angle, actual.angle) << "angle field has a wrong value after being written to a file and then read";
    }

    // OutputParams helper functions
    OutputParams createOutputParamsFixed()
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
        outputParams.add_angle("dirTest11", 56, 57.0, 58, 59, 60, false);
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

        return outputParams;
    }

    OutputParams createOutputParamsRandom()
    {
        OutputParams outputParams(TestUtils::random_string(TestUtils::randomStringLength));
        for (int i = 0; i < TestUtils::randomArrayMinimalSize + rand() % TestUtils::randomArraySize; i++)
        {
            outputParams.add_XY_plane(TestUtils::random_string(TestUtils::randomStringLength), rand(), rand(), rand(), rand(), rand(), ((rand() & 1) == 1));
        }
        for (int i = 0; i < TestUtils::randomArrayMinimalSize + rand() % TestUtils::randomArraySize; i++)
        {
            outputParams.add_XZ_plane(TestUtils::random_string(TestUtils::randomStringLength), rand(), rand(), rand(), rand(), rand(), ((rand() & 1) == 1));
        }
        for (int i = 0; i < TestUtils::randomArrayMinimalSize + rand() % TestUtils::randomArraySize; i++)
        {
            outputParams.add_YZ_plane(TestUtils::random_string(TestUtils::randomStringLength), rand(), rand(), rand(), rand(), rand(), ((rand() & 1) == 1));
        }

        std::uniform_real_distribution<double> unif(TestUtils::randomDoubleLowerBound, TestUtils::randomDoubleUpperBound);
        std::default_random_engine re;

        for (int i = 0; i < TestUtils::randomArrayMinimalSize + rand() % TestUtils::randomArraySize; i++)
        {
            outputParams.add_angle(TestUtils::random_string(TestUtils::randomStringLength), rand(), unif(re), rand(), rand(), rand(), ((rand() & 1) == 1));
        }

        for (int i = 0; i < TestUtils::randomArrayMinimalSize + rand() % TestUtils::randomArraySize; i++)
        {
            outputParams.add_YZ_plane_at_angle(TestUtils::random_string(TestUtils::randomStringLength), unif(re), unif(re), rand(), rand(), rand(), rand(), ((rand() & 1) == 1));
        }

        for (int i = 0; i < TestUtils::randomArrayMinimalSize + rand() % TestUtils::randomArraySize; i++)
        {
            outputParams.add_volume(TestUtils::random_string(TestUtils::randomStringLength), rand(), rand(), rand(), rand(), ((rand() & 1) == 1));
        }
        return outputParams;
    }

    void checkAllFields(OrthoPlaneParams &expected, OrthoPlaneParams &actual)
    {
        ASSERT_EQ(expected.name_root, actual.name_root) << "name_root field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.repeat, actual.repeat) << "repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.cutAt, actual.cutAt) << "cutAt field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Q_output_len, actual.Q_output_len) << "Q_output_len field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.start_at_step, actual.start_at_step) << "start_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.end_at_step, actual.end_at_step) << "end_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.use_half_float, actual.use_half_float) << "use_half_float field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
    }

    void checkAllFields(VolumeParams &expected, VolumeParams &actual)
    {
        ASSERT_EQ(expected.name_root, actual.name_root) << "name_root field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.repeat, actual.repeat) << "repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Q_output_len, actual.Q_output_len) << "Q_output_len field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.start_at_step, actual.start_at_step) << "start_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.end_at_step, actual.end_at_step) << "end_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.use_half_float, actual.use_half_float) << "use_half_float field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
    }

    void checkAllFields(AngleParams &expected, AngleParams &actual)
    {
        ASSERT_EQ(expected.name_root, actual.name_root) << "name_root field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.repeat, actual.repeat) << "repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.degrees, actual.degrees) << "degrees field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Q_output_len, actual.Q_output_len) << "Q_output_len field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.start_at_step, actual.start_at_step) << "start_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.end_at_step, actual.end_at_step) << "end_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.use_half_float, actual.use_half_float) << "use_half_float field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
    }

    void checkAllFields(PlaneAtAngleParams &expected, PlaneAtAngleParams &actual)
    {
        ASSERT_EQ(expected.name_root, actual.name_root) << "name_root field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.degrees, actual.degrees) << "degrees field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.tolerance, actual.tolerance) << "tolerance field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.cutAt, actual.cutAt) << "cutAt field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Q_output_len, actual.Q_output_len) << "Q_output_len field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.start_at_step, actual.start_at_step) << "start_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.end_at_step, actual.end_at_step) << "end_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.use_half_float, actual.use_half_float) << "use_half_float field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
    }

    void checkAllFields(SectorParams &expected, SectorParams &actual)
    {
        ASSERT_EQ(expected.name_root, actual.name_root) << "name_root field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.repeat, actual.repeat) << "repeat field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.angle_infront_blade, actual.angle_infront_blade) << "angle_infront_blade field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.angle_behind_blade, actual.angle_behind_blade) << "angle_behind_blade field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Q_output_len, actual.Q_output_len) << "Q_output_len field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.start_at_step, actual.start_at_step) << "start_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.end_at_step, actual.end_at_step) << "end_at_step field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.use_half_float, actual.use_half_float) << "use_half_float field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.QDataType, actual.QDataType) << "QDataType field has a wrong value after being written to a file and then read";
    }

    void checkAllFields(OutputParams &expected, OutputParams &actual)
    {
        ASSERT_EQ(expected.outputRootDir, actual.outputRootDir) << "outputRootDir field has a wrong value after being written to a file and then read";

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

    void generateTestDataHeader(std::ostream &str, std::string fname)
    {
        str << "namespace TestUtils {\n";
        str << "    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>\n";
        str << "    void " << fname << "(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu) {\n";
        str << "        QVec<T, QVecSize> qTmp;\n";
    }

    void generateTestDataFooter(std::ostream &str)
    {
        str << "    }\n}\n";
    }

#if WITH_MPI == 1
    void generateTestDataHeaderMpi(std::ostream &str, std::string fname)
    {
        str << "namespace TestUtils {\n";
        str << "    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, MemoryLayoutType MemoryLayoutHalo>\n";
        str << "    void " << fname << "(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, ComputeUnitBase<T, QVecSize, MemoryLayoutHalo> **halos) {\n";
        str << "        QVec<T, QVecSize> qTmp;\n";
    }
#endif
}
