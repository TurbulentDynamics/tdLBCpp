//
//  StreamingNieveTest.cpp
//  StreamingNieveTest
//
//  Unit tests for StreamingNieve
//

#include <cstdio>

#include "gtest/gtest.h"

#include "Header.h"
#include "Params/Running.hpp"
#include "Params/Checkpoint.hpp"
#include "Params/OutputParams.hpp"
#include "ComputeUnit.hpp"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"
#include "StreamingNieveTest.hpp"

template <typename T, int QVecSize>
void fillForTest(ComputeUnit<T, QVecSize> cu)
{

    if (cu.xg > 99 || cu.yg > 99 || cu.zg > 99)
    {
        std::cout << "Size too large for testing" << std::endl;
        exit(1);
    }
#if WITH_GPU == 1
    setToZero<<<numBlocks, threadsPerBlock>>>(devN, devF, xg, yg, zg, QVecSize);
#else
    for (tNi i = 0; i < cu.xg; i++)
    {
        for (tNi j = 0; j < cu.yg; j++)
        {
            for (tNi k = 0; k < cu.zg; k++)
            {

                QVec<unsigned long int, QVecSize> qTmp;

                for (unsigned long int l = 0; l < QVecSize; l++)
                {
                    qTmp.q[l] = i * 1000000 + j * 10000 + k * 100 + l;
                }
                cu.Q[cu.index(i, j, k)] = qTmp;

                cu.F[cu.index(i, j, k)].x = 0;
                cu.F[cu.index(i, j, k)].y = 1;
                cu.F[cu.index(i, j, k)].z = 2;

                cu.Nu[cu.index(i, j, k)] = 1;
                cu.O[cu.index(i, j, k)] = true;
            }
        }
    }
#endif
};

template <typename T, int QVecSize>
void generateTestData(ComputeUnit<T, QVecSize> cu)
{
    std::cerr << "namespace TestUtils {\n";
    std::cerr << "    template <typename T, int QVecSize>\n";
    std::cerr << "    void fillExpectedComputeUnitValues(ComputeUnit<T, QVecSize> cu) {\n";
    std::cerr << "        QVec<T, QVecSize> qTmp;\n";
    for (tNi i = 0; i < cu.xg; i++)
    {
        for (tNi j = 0; j < cu.yg; j++)
        {
            for (tNi k = 0; k < cu.zg; k++)
            {
                for (unsigned long int l = 0; l < QVecSize; l++)
                {
                    if ((l > 0) && (l % 8 == 0))
                    {
                        std::cerr << "\n       ";
                    }
                    if (l == 0)
                    {
                        std::cerr << "       ";
                    }
                    std::cerr << " qTmp.q[" << l << "] = " << cu.Q[cu.index(i, j, k)].q[l] << ";";
                }
                std::cerr << "\n        cu.Q[cu.index(" << i << ", " << j << ", " << k << ")] = qTmp;\n";
            }
        }
    }
    std::cerr << "    }\n};\n";
}

template <typename T, int QVecSize>
void testStream(ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree, ComputeUnit<T, QVecSize> actual)
{
    ComputeUnit<T, QVecSize> expected = ComputeUnit<T, QVecSize>(cuParams, flow, outputTree);
    TestUtils::fillExpectedComputeUnitValues(expected);
    for (tNi i = 0; i < actual.xg; i++)
    {
        for (tNi j = 0; j < actual.yg; j++)
        {
            for (tNi k = 0; k < actual.zg; k++)
            {
                for (unsigned long int l = 0; l < QVecSize; l++)
                {
                    ASSERT_EQ(actual.Q[actual.index(i, j, k)].q[l], expected.Q[expected.index(i, j, k)].q[l])
                        << "value for Q doesn't match at " << i << ", " << j << ", " << k << ", " << l << " index, actual: "
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
    checkpointParams.checkpoint_root_dir = TestUtils::getTempFilename(testing::TempDir(), "StreamingNieveTestRoot");
    OutputParams outputParams(TestUtils::getTempFilename(testing::TempDir(), "StreamingNieveTestOutput"));
    DiskOutputTree diskOutputTree(checkpointParams, outputParams);
    GridParams gridParams = ParamsCommon::createGridParamsFixed();
    RunningParams runningParams = ParamsCommon::createRunningParamsFixed();
    diskOutputTree.setParams(cuParams, gridParams, flow, runningParams, outputParams, checkpointParams);
    // if cu parameters change then StreamingNieveTest.hpp needs to be recreated
    cuParams.x = 3;
    cuParams.y = 3;
    cuParams.z = 3;
    cuParams.ghost = 1;

    ComputeUnit<unsigned long, QLen::D3Q19> lb2 = ComputeUnit<unsigned long, QLen::D3Q19>(cuParams, flow, diskOutputTree);

    fillForTest(lb2);
    lb2.streamingNieve();
    //generateTestData(lb2);
    testStream(cuParams, flow, diskOutputTree, lb2);
}