//
//  ComputeUnitMemoryTest.cpp
//  ComputeUnitMemoryTest
//
//  Unit tests for different memory layouts
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

class ComputeUnitMemoryTest : public ::testing::Test
{
protected:
    ComputeUnitParams cuParams;
    CheckpointParams checkpointParams;
    FlowParams<unsigned long> flow;
    OutputParams outputParams;
    GridParams gridParams;
    RunningParams runningParams;

    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout,
              typename std::enable_if<
                  MemoryLayout == MemoryLayoutIJKL,
                  bool>::type = true>
    void compareValuesAtIndex(std::string tag, ComputeUnitBase<T, QVecSize, MemoryLayout> &actual, T valueAtIJKL,
                              tNi i, tNi j, tNi k, unsigned long int l)
    {
        T expectedValue = actual.Q.q[actual.index(i, j, k) * QVecSize + l];
        ASSERT_EQ(valueAtIJKL, expectedValue)
            << tag << " : value for Q doesn't comply with memory layout chosen at " << i << ", " << j << ", " << k << ", " << l << " index, actual: "
            << valueAtIJKL
            << " != expected for memory layout " << expectedValue;
    }

    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout,
              typename std::enable_if<
                  MemoryLayout == MemoryLayoutLIJK,
                  bool>::type = true>
    void compareValuesAtIndex(std::string tag, ComputeUnitBase<T, QVecSize, MemoryLayout> &actual, T valueAtIJKL,
                              tNi i, tNi j, tNi k, unsigned long int l)
    {
        T expectedValue = actual.Q.q[l * actual.xg * actual.yg * actual.zg + actual.index(i, j, k)];
        ASSERT_EQ(valueAtIJKL, expectedValue)
            << tag << " : value for Q doesn't comply with memory layout chosen at " << i << ", " << j << ", " << k << ", " << l << " index, actual: "
            << valueAtIJKL
            << " != expected for memory layout " << expectedValue;
    }

    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
    void testMemoryLayout(std::string tag, ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree, ComputeUnitBase<T, QVecSize, MemoryLayout> &actual)
    {
        for (tNi i = 0; i < actual.xg; i++)
        {
            for (tNi j = 0; j < actual.yg; j++)
            {
                for (tNi k = 0; k < actual.zg; k++)
                {
                    for (unsigned long int l = 0; l < QVecSize; l++)
                    {
                        T valueAtIJKL = actual.Q[actual.index(i, j, k)].q[l];
                        compareValuesAtIndex(tag, actual, valueAtIJKL,
                                             i, j, k, l);
                    }
                }
            }
        }
    }

public:
    ComputeUnitMemoryTest() : cuParams(ParamsCommon::createComputeUnitParamsFixed()),
                              checkpointParams(ParamsCommon::createCheckpointParamsFixed()),
                              outputParams(TestUtils::getTempFilename(testing::TempDir(), "ComputeUnitMemoryTestOutput")),
                              gridParams(ParamsCommon::createGridParamsFixed()),
                              runningParams(ParamsCommon::createRunningParamsFixed())
    {
        checkpointParams.checkpointWriteRootDir = TestUtils::getTempFilename(testing::TempDir(), "ComputeUnitMemoryTestRoot");
        cuParams.x = 3;
        cuParams.y = 3;
        cuParams.z = 3;
        cuParams.ghost = 1;
    }
};

TEST_F(ComputeUnitMemoryTest, ComputeUnitMemoryIJKLTest)
{
    DiskOutputTree diskOutputTree = DiskOutputTree(outputParams, checkpointParams);
    diskOutputTree.setParams(cuParams, gridParams.getJson(), flow.getJson(), runningParams.getJson(), outputParams.getJson(), checkpointParams.getJson());

    auto lb = ComputeUnit<unsigned long, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple, CPU>(cuParams, flow, diskOutputTree);

    ParamsCommon::fillForTest<unsigned long, QLen::D3Q19, MemoryLayoutIJKL>(lb);

    testMemoryLayout("IJKL", cuParams, flow, diskOutputTree, lb);
}

TEST_F(ComputeUnitMemoryTest, ComputeUnitMemoryLIJKTest)
{
    DiskOutputTree diskOutputTree = DiskOutputTree(outputParams, checkpointParams);
    diskOutputTree.setParams(cuParams, gridParams.getJson(), flow.getJson(), runningParams.getJson(), outputParams.getJson(), checkpointParams.getJson());

    auto lb = ComputeUnit<unsigned long, QLen::D3Q19, MemoryLayoutLIJK, EgglesSomers, Simple, CPU>(cuParams, flow, diskOutputTree);

    ParamsCommon::fillForTest<unsigned long, QLen::D3Q19, MemoryLayoutLIJK>(lb);

    testMemoryLayout("LIJK", cuParams, flow, diskOutputTree, lb);
}