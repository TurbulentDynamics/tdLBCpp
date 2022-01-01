//
//  StreamingNieveTest.cpp
//  StreamingNieveTest
//
//  Unit tests for StreamingNieve
//

#include <cstdio>
#include <fstream>

#include "gtest/gtest.h"

#undef WITH_GPU

#include "Header.h"
#include "Params/RunningParams.hpp"
#include "Params/CheckpointParams.hpp"
#include "Params/OutputParams.hpp"
#include "ComputeUnit.hpp"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"
#include "StreamingSimplePushTest.hpp"

void createGpuUnitsExecutePush(unsigned long *q, unsigned long *qLijk, ComputeUnitParams cuParams, FlowParams<unsigned long> flow, DiskOutputTree diskOutputTree);

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void testStream(std::string tag, ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree, ComputeUnitBase<T, QVecSize, MemoryLayout> &actual)
{
    ComputeUnit<T, QVecSize, MemoryLayout, EgglesSomers, Simple, CPU> expected(cuParams, flow, outputTree);
    TestUtils::fillExpectedComputeUnitValuesTestStreamingNievePush(expected);
    for (tNi i = 0; i < actual.xg; i++)
    {
        for (tNi j = 0; j < actual.yg; j++)
        {
            for (tNi k = 0; k < actual.zg; k++)
            {
                for (unsigned long int l = 0; l < QVecSize; l++)
                {
                    //std::cerr << "ind: i,j,k,l = " << i << ", " << j << ", " << k << ", " << l << std::endl;
                    //std::cerr << "act: " << actual.Q[actual.index(i, j, k)].q[l] << ", exp: " << expected.Q[expected.index(i, j, k)].q[l] << std::endl;
                    ASSERT_EQ(actual.Q[actual.index(i, j, k)].q[l], expected.Q[expected.index(i, j, k)].q[l])
                        << tag << " : value for Q doesn't match at " << i << ", " << j << ", " << k << ", " << l << " index, actual: "
                        << actual.Q[actual.index(i, j, k)].q[l]
                        << " != expected " << expected.Q[expected.index(i, j, k)].q[l];
                }
            }
        }
    }
}

TEST(StreamingNieveGpuTest, StreamingNieveGpuValidTest)
{
    // Prepare test parameters for ComputeUnit construction
    ComputeUnitParams cuParams = ParamsCommon::createComputeUnitParamsFixed();
    FlowParams<unsigned long> flow;
    CheckpointParams checkpointParams = ParamsCommon::createCheckpointParamsFixed();
    checkpointParams.checkpointWriteRootDir = TestUtils::getTempFilename(testing::TempDir(), "StreamingNieveTestRoot");
    OutputParams outputParams(TestUtils::getTempFilename(testing::TempDir(), "StreamingNieveTestOutput"));
    DiskOutputTree diskOutputTree(outputParams, checkpointParams);
    GridParams gridParams = ParamsCommon::createGridParamsFixed();
    RunningParams runningParams = ParamsCommon::createRunningParamsFixed();
    diskOutputTree.setParams(cuParams, gridParams.getJson(), flow.getJson(), runningParams.getJson(), outputParams.getJson(), checkpointParams.getJson());
    // if cu parameters change then StreamingNieveTest.hpp needs to be recreated
    cuParams.x = 3;
    cuParams.y = 3;
    cuParams.z = 3;
    cuParams.ghost = 1;

    ComputeUnit<unsigned long, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple, CPU> lb2(cuParams, flow, diskOutputTree);
    ComputeUnit<unsigned long, QLen::D3Q19, MemoryLayoutLIJK, EgglesSomers, Simple, CPU> lb2lijk(cuParams, flow, diskOutputTree);

    createGpuUnitsExecutePush(lb2.Q.q, lb2lijk.Q.q, cuParams, flow, diskOutputTree);

    testStream("IJKL", cuParams, flow, diskOutputTree, lb2);

    testStream("LIJK", cuParams, flow, diskOutputTree, lb2lijk);
}
