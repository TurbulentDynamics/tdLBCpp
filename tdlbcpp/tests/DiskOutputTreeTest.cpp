//
//  DiskOutputTreeTest.cpp
//  DiskOutputTreeTest
//
//  Unit tests for DiskOutputTree
//

#include <cstdio>
#include <cstdlib>
#include <random>

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/Running.hpp"
#include "Params/Checkpoint.hpp"
#include "Params/OutputParams.hpp"
#include "Params/BinFile.hpp"
#include "DiskOutputTree.h"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"

class DiskOutputTreeTests : public ::testing::Test
{
protected:
    std::string filename;
    const std::string diskOutputTreeTestsFolderRoot = "diskOutputTreeTestsFolderRoot";
    const std::string diskOutputTreeTestsFolderCheckpoint = "diskOutputTreeTestsFolderCheckpoint";
    std::string rootDirDiskOutputTree;
    std::string checkpointDirOfDiskOutputTree;

    void checkAllFields(DiskOutputTree &expected, DiskOutputTree &actual)
    {
        // TODO: check all loaded data recursively
    }

public:
    DiskOutputTreeTests()
    {
        rootDirDiskOutputTree = TestUtils::joinPath(testing::TempDir(), diskOutputTreeTestsFolderRoot);
        checkpointDirOfDiskOutputTree = TestUtils::joinPath(testing::TempDir(), diskOutputTreeTestsFolderCheckpoint);
        filename = TestUtils::getTempFilename(testing::TempDir(), "_to_delete");
    }
    ~DiskOutputTreeTests()
    {
        TestUtils::removeTempFile(filename + ".json");
    }
};

TEST_F(DiskOutputTreeTests, DiskOutputTreeWriteReadValidTest)
{
    CheckpointParams checkpointParams = ParamsCommon::createCheckpointParamsFixed();
    checkpointParams.checkpoint_root_dir = checkpointDirOfDiskOutputTree;
    OutputParams outputParams(rootDirDiskOutputTree);
    DiskOutputTree diskOutputTree(checkpointParams, outputParams);

    ComputeUnitParams computeUnitParams = ParamsCommon::createComputeUnitParamsFixed();
    GridParams gridParams = ParamsCommon::createGridParamsFixed();
    FlowParams<double> flowParams = ParamsCommon::createFlowParamsFixed();
    RunningParams runningParams = ParamsCommon::createRunningParamsFixed();
    diskOutputTree.setParams(computeUnitParams, gridParams, flowParams, runningParams, outputParams, checkpointParams);

    BinFileParams binFileParams;
    binFileParams.filePath = filename;
    RunningParams runParams;
    diskOutputTree.writeAllParamsJson(binFileParams, runParams);
    std::cerr << filename << std::endl;

    CheckpointParams checkpointParamsNew = ParamsCommon::createCheckpointParamsFixed();
    OutputParams outputParamsNew(rootDirDiskOutputTree + "test2");
    DiskOutputTree diskOutputTreeRead(checkpointParamsNew, outputParamsNew);
    diskOutputTreeRead.readAllParamsJson(filename + ".json", binFileParams, runParams);

    checkAllFields(diskOutputTree, diskOutputTreeRead);
    ASSERT_EQ(diskOutputTree.getCheckpointDirName(runningParams, false),
              diskOutputTreeRead.getCheckpointDirName(runningParams, false))
        << "getCheckpointDirName() function returns wrong value after being written to a file and then read";

    ASSERT_EQ(diskOutputTree.getCheckpointDirName(runningParams, true),
              diskOutputTreeRead.getCheckpointDirName(runningParams, true))
        << "getCheckpointDirName() function returns wrong value after being written to a file and then read";

    ASSERT_EQ(diskOutputTree.getCheckpointFilePath("test1", "test2", "test3"),
              diskOutputTreeRead.getCheckpointFilePath("test1", "test2", "test3"))
        << "getCheckpointFilePath() function returns wrong value after being written to a file and then read";

    std::string dirName = TestUtils::random_string(TestUtils::randomStringLength);
    std::string unit_name = TestUtils::random_string(TestUtils::randomStringLength);
    std::string matrix = TestUtils::random_string(TestUtils::randomStringLength);
    ASSERT_EQ(diskOutputTree.getCheckpointFilePath(dirName, unit_name, matrix),
              diskOutputTreeRead.getCheckpointFilePath(dirName, unit_name, matrix))
        << "getCheckpointFilePath() function returns wrong value after being written to a file and then read";

    ASSERT_EQ(diskOutputTree.formatCUid(),
              diskOutputTreeRead.formatCUid())
        << "formatCUid() function returns wrong value after being written to a file and then read";

    std::string randomDir = TestUtils::random_string(TestUtils::randomStringLength);
    ASSERT_EQ(diskOutputTree.formatQVecBinFileNamePath(randomDir),
              diskOutputTreeRead.formatQVecBinFileNamePath(randomDir))
        << "formatQVecBinFileNamePath() function returns wrong value after being written to a file and then read";

    ASSERT_EQ(diskOutputTree.formatF3BinFileNamePath(randomDir),
              diskOutputTreeRead.formatF3BinFileNamePath(randomDir))
        << "formatF3BinFileNamePath() function returns wrong value after being written to a file and then read";
}
