//
//  StreamingNieveTest.cpp
//  StreamingNieveTest
//
//  Unit tests for StreamingNieve
//

#include <cstdio>
#include <fstream>

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/RunningParams.hpp"
#include "Params/CheckpointParams.hpp"
#include "Params/OutputParams.hpp"
#include "ComputeUnit.hpp"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"
#include "StreamingSimpleTest.hpp"
#include "StreamingSimplePushTest.hpp"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void testStream(bool push, std::string tag, ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree, ComputeUnitBase<T, QVecSize, MemoryLayout> &actual)
{
    ComputeUnit<T, QVecSize, MemoryLayout, EgglesSomers, Simple, CPU> expected(cuParams, flow, outputTree);
    if (push)
    {
        TestUtils::fillExpectedComputeUnitValuesTestStreamingNievePush(expected);
    }
    else
    {
        TestUtils::fillExpectedComputeUnitValuesTestStreamingNieve(expected);
    }
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

TEST(StreamingNieveTest, StreamingNieveValidTest)
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

    ParamsCommon::fillForTest(lb2);
    lb2.streamingPull();
    if (std::getenv("GENERATE_STREAMING_NIEVE_TEST_HPP"))
    {
        std::string headerPath = std::getenv("GENERATE_STREAMING_NIEVE_TEST_HPP");
        std::cerr << "Writing to headerPath: " << headerPath << std::endl;
        ParamsCommon::generateTestData(lb2, headerPath, "TestStreamingNieve");
    }
    testStream(false, "IJKL", cuParams, flow, diskOutputTree, lb2);

    ParamsCommon::fillForTest(lb2lijk);
    lb2lijk.streamingPull();
    testStream(false, "LIJK", cuParams, flow, diskOutputTree, lb2lijk);
}

TEST(StreamingNieveTest, StreamingNievePushValidTest)
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

    ParamsCommon::fillForTest(lb2);
    lb2.streamingPush();
    if (std::getenv("GENERATE_STREAMING_NIEVE_PUSH_TEST_HPP"))
    {
        std::string headerPath = std::getenv("GENERATE_STREAMING_NIEVE_PUSH_TEST_HPP");
        std::cerr << "Writing to headerPath: " << headerPath << std::endl;
        ParamsCommon::generateTestData(lb2, headerPath, "TestStreamingNievePush");
    }
    testStream(true, "IJKL", cuParams, flow, diskOutputTree, lb2);

    ParamsCommon::fillForTest(lb2lijk);
    lb2lijk.streamingPush();
    testStream(true, "LIJK", cuParams, flow, diskOutputTree, lb2lijk);
}
