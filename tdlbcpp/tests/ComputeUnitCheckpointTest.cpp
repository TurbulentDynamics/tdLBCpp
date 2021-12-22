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
#include "Params/RunningParams.hpp"
#include "Params/CheckpointParams.hpp"
#include "Params/OutputParams.hpp"
#include "Params/BinFileParams.hpp"
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
    using ComputeUnitT = ComputeUnit<double, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple, CPU>;
    std::string unitName = TestUtils::random_string(10);
    CheckpointParams checkpointParams = ParamsCommon::createCheckpointParamsFixed();
    checkpointParams.checkpointWriteRootDir = checkpointTestsFolderCheckpointFull;
    TestUtils::createDir(checkpointTestsFolderCheckpointFull);
    OutputParams outputParams(checkpointTestsFolderCheckpointFull);
    DiskOutputTree diskOutputTree(outputParams, checkpointParams);

    ComputeUnitParams computeUnitParams = ParamsCommon::createComputeUnitParamsFixed();
    GridParams gridParams = ParamsCommon::createGridParamsFixed();
    FlowParams<double> flowParams = ParamsCommon::createFlowParamsWithRandomValues<double>();
    RunningParams runningParams = ParamsCommon::createRunningParamsFixed();
    diskOutputTree.setParams(computeUnitParams, gridParams.getJson(), flowParams.getJson(), runningParams.getJson(), outputParams.getJson(), checkpointParams.getJson());

    BinFileParams binFileParams;
    binFileParams.filePath = filename;

    ComputeUnitParams cuParams = ParamsCommon::createComputeUnitParamsFixed();

    ComputeUnitT lb2(cuParams, flowParams, diskOutputTree);

    lb2.checkpoint_write(unitName, runningParams);

    ComputeUnitT lb2read(cuParams, flowParams, diskOutputTree);

    lb2read.checkpoint_read(diskOutputTree.getCheckpointDirName(runningParams), unitName);

    ParamsCommon::checkAllFields(lb2, lb2read);
}
