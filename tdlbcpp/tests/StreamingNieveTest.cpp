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
#include "Params/Running.hpp"
#include "Params/Checkpoint.hpp"
#include "Params/OutputParams.hpp"
#include "ComputeUnit.hpp"

#include "tdlbcpp/tests/utils.hpp"
#include "tdlbcpp/tests/Params/ParamsCommon.hpp"
#include "StreamingNieveTest.hpp"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void generateTestData(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, std::string headerPath)
{
    std::ofstream hdr(headerPath);
    hdr << "namespace TestUtils {\n";
    hdr << "    template <typename T, int QVecSize>\n";
    hdr << "    void fillExpectedComputeUnitValues(ComputeUnit<T, QVecSize> cu) {\n";
    hdr << "        QVec<T, QVecSize> qTmp;\n";
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
                        hdr << "\n       ";
                    }
                    if (l == 0)
                    {
                        hdr << "       ";
                    }
                    hdr << " qTmp.q[" << l << "] = " << cu.Q[cu.index(i, j, k)].q[l] << ";";
                }
                hdr << "\n        cu.Q[cu.index(" << i << ", " << j << ", " << k << ")] = qTmp;\n";
            }
        }
    }
    hdr << "    }\n};\n";
    hdr.close();
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void testStream(std::string tag, ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree, ComputeUnitBase<T, QVecSize, MemoryLayout> &actual)
{
    ComputeUnit<T, QVecSize, MemoryLayout, EgglesSomers, Simple> expected(cuParams, flow, outputTree);
    TestUtils::fillExpectedComputeUnitValues(expected);
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

    ComputeUnit<unsigned long, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple> lb2(cuParams, flow, diskOutputTree);
    ComputeUnit<unsigned long, QLen::D3Q19, MemoryLayoutLIJK, EgglesSomers, Simple> lb2lijk(cuParams, flow, diskOutputTree);

    ParamsCommon::fillForTest(lb2);
    lb2.streaming();
    if (std::getenv("GENERATE_STREAMING_NIEVE_TEST_HPP"))
    {
        std::string headerPath = std::getenv("GENERATE_STREAMING_NIEVE_TEST_HPP");
        generateTestData(lb2, headerPath);
    }
    testStream("IJKL", cuParams, flow, diskOutputTree, lb2);

    ParamsCommon::fillForTest(lb2lijk);
    lb2lijk.streaming();
    testStream("LIJK", cuParams, flow, diskOutputTree, lb2lijk);
}