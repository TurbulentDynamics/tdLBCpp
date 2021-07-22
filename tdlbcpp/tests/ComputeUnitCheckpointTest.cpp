//
//  ComputeUnitCheckpointTest.cpp
//  ComputeUnitCheckpointTest
//
//  Unit tests for ComputeUnit checkpoint code
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
#include "ComputeUnit.hpp"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"

class ComputeUnitCheckpointTests : public ::testing::Test
{
protected:
    std::string filename;
    const std::string checkpointTestsFolderOutput = "checkpointTestsFolderOutput";
    const std::string checkpointTestsFolderCheckpoint = "checkpointTestsFolderCheckpoint";
    std::string checkpointTestsFolderOutputFull;
    std::string checkpointTestsFolderCheckpointFull;

public:
    ComputeUnitCheckpointTests()
    {
        checkpointTestsFolderOutputFull = TestUtils::joinPath(testing::TempDir(), checkpointTestsFolderOutput);
        checkpointTestsFolderCheckpointFull = TestUtils::joinPath(testing::TempDir(), checkpointTestsFolderCheckpoint);
        filename = TestUtils::getTempFilename(testing::TempDir(), "_to_delete");
    }
    ~ComputeUnitCheckpointTests()
    {
        TestUtils::removeTempFile(filename + ".json");
    }
};

TEST_F(ComputeUnitCheckpointTests, ComputeUnitCheckpointWriteReadValid)
{
    std::string unitName = TestUtils::random_string(10);
    CheckpointParams checkpointParams = ParamsCommon::createCheckpointParamsFixed();
    checkpointParams.checkpoint_root_dir = checkpointTestsFolderCheckpointFull;
    OutputParams outputParams(checkpointTestsFolderCheckpointFull);
    DiskOutputTree diskOutputTree(checkpointParams, outputParams);

    ComputeUnitParams computeUnitParams = ParamsCommon::createComputeUnitParamsFixed();
    GridParams gridParams = ParamsCommon::createGridParamsFixed();
    FlowParams<double> flowParams = ParamsCommon::createFlowParamsWithRandomValues<double>();
    RunningParams runningParams = ParamsCommon::createRunningParamsFixed();
    diskOutputTree.setParams(computeUnitParams, gridParams, flowParams, runningParams, outputParams, checkpointParams);

    BinFileParams binFileParams;
    binFileParams.filePath = filename;

    ComputeUnitParams cuParams = ParamsCommon::createComputeUnitParamsFixed();

    auto lb2 = ComputeUnit<double, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple>(cuParams, flowParams, diskOutputTree);

    lb2.checkpoint_write(unitName, runningParams);

    auto lb2read = ComputeUnit<double, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple>(cuParams, flowParams, diskOutputTree);

    lb2read.checkpoint_read(diskOutputTree.getCheckpointDirName(runningParams), unitName);

    ParamsCommon::checkAllFields(lb2, lb2read);
}
