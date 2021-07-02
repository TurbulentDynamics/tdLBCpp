#include <random>

#include "Params/Checkpoint.hpp"
#include "Params/BinFile.hpp"
#include "Params/ComputeUnitParams.hpp"
#include "Params/Flow.hpp"
#include "Params/Grid.hpp"
#include "Params/Running.hpp"
#include "Params/OutputParams.hpp"

#include "tdlbcpp/tests/utils.hpp"

namespace ParamsCommon
{
    // CheckpointParams helper methods
    CheckpointParams createCheckpointParamsFixed();
    CheckpointParams createCheckpointParamsRandom();
    void checkAllFields(CheckpointParams &expected, CheckpointParams &actual);

    // BinFileParams helper methods
    BinFileParams createBinFileParamsFixed();
    BinFileParams createBinFileParamsRandom();
    void checkAllFields(BinFileParams &expected, BinFileParams &actual);

    // ComputeUnitParams helper methods
    ComputeUnitParams createComputeUnitParamsFixed();
    ComputeUnitParams createComputeUnitParamsRandom();
    void checkAllFields(ComputeUnitParams &expected, ComputeUnitParams &actual);

    // FlowParams helper methods
    FlowParams<double> createFlowParamsFixed();
    template <typename T>
    FlowParams<T> createFlowParamsWithRandomValues()
    {
        FlowParams<T> flowParams;

        std::uniform_real_distribution<double> unif(TestUtils::randomDoubleLowerBound, TestUtils::randomDoubleUpperBound);
        std::default_random_engine re;

        flowParams.initialRho = (T)unif(re);
        flowParams.reMNonDimensional = (T)unif(re);
        flowParams.uav = (T)unif(re);
        flowParams.cs0 = (T)unif(re);
        flowParams.g3 = (T)unif(re);
        flowParams.nu = (T)unif(re);
        flowParams.fx0 = (T)unif(re);
        flowParams.Re_m = (T)unif(re);
        flowParams.Re_f = (T)unif(re);
        flowParams.uf = (T)unif(re);
        flowParams.alpha = (T)unif(re);
        flowParams.beta = (T)unif(re);
        flowParams.useLES = (rand() & 1) == 1;
        flowParams.collision = TestUtils::random_string(TestUtils::randomStringLength);
        flowParams.streaming = TestUtils::random_string(TestUtils::randomStringLength);

        return flowParams;
    }

    template <typename T>
    void checkAllFields(FlowParams<T> &expected, FlowParams<T> &actual)
    {
        ASSERT_EQ(expected.initialRho, actual.initialRho) << "initialRho field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.reMNonDimensional, actual.reMNonDimensional) << "reMNonDimensional field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.uav, actual.uav) << "uav field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.cs0, actual.cs0) << "cs0 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.g3, actual.g3) << "g3 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.nu, actual.nu) << "nu field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.fx0, actual.fx0) << "fx0 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Re_m, actual.Re_m) << "Re_m field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.Re_f, actual.Re_f) << "Re_f field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.uf, actual.uf) << "uf field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.alpha, actual.alpha) << "alpha field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.beta, actual.beta) << "beta field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.useLES, actual.useLES) << "useLES field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.collision, actual.collision) << "collision field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.streaming, actual.streaming) << "streaming field has a wrong value after being written to a file and then read";
    }

    // GridParams helper methods
    GridParams createGridParamsFixed();
    GridParams createGridParamsRandom();
    void checkAllFields(GridParams &expected, GridParams &actual);

    // RunningParams helper functions
    RunningParams createRunningParamsFixed();
    RunningParams createRunningParamsRandom();
    void checkAllFields(RunningParams &expected, RunningParams &actual);

    // OutputParams helper functions
    OutputParams createOutputParamsFixed();
    OutputParams createOutputParamsRandom();
    void checkAllFields(OutputParams &expected, OutputParams &actual);
}