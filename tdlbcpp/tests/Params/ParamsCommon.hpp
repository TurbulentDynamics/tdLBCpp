#include <random>

#include "Params/Checkpoint.hpp"
#include "Params/BinFile.hpp"
#include "Params/ComputeUnitParams.hpp"
#include "Params/Flow.hpp"
#include "Params/Grid.hpp"
#include "Params/Running.hpp"
#include "Params/OutputParams.hpp"
#include "ComputeUnit.hpp"

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

    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
    void checkAllFields(ComputeUnitBase<T, QVecSize, MemoryLayout> &expected, ComputeUnitBase<T, QVecSize, MemoryLayout> &actual)
    {
        checkAllFields(expected.flow, actual.flow);
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

        ASSERT_EQ(expected.xg, actual.xg) << "xg field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.yg, actual.yg) << "yg field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.zg, actual.zg) << "zg field has a wrong value after being written to a file and then read";

        ASSERT_EQ(expected.xg0, actual.xg0) << "xg0 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.yg0, actual.yg0) << "yg0 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.zg0, actual.zg0) << "zg0 field has a wrong value after being written to a file and then read";

        ASSERT_EQ(expected.xg1, actual.xg1) << "xg1 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.yg1, actual.yg1) << "yg1 field has a wrong value after being written to a file and then read";
        ASSERT_EQ(expected.zg1, actual.zg1) << "zg1 field has a wrong value after being written to a file and then read";

        ASSERT_EQ(expected.size, actual.size) << "size field has a wrong value after being written to a file and then read";

        for (size_t i = 0; i < expected.size; i++)
        {
            ASSERT_EQ(expected.F[i].x, actual.F[i].x) << "F[" << i << "].x field has a wrong value after being written to a file and then read";
            ASSERT_EQ(expected.F[i].y, actual.F[i].y) << "F[" << i << "].y field has a wrong value after being written to a file and then read";
            ASSERT_EQ(expected.F[i].z, actual.F[i].z) << "F[" << i << "].z field has a wrong value after being written to a file and then read";
            for (int j = 0; j < QVecSize; j++)
            {
                ASSERT_EQ(expected.Q[i].q[j], actual.Q[i].q[j]) << "Q[" << i << "].q[" << j << "] field has a wrong value after being written to a file and then read";
            }
            ASSERT_EQ(expected.Nu[i], actual.Nu[i]) << "Nu[" << i << "] field has a wrong value after being written to a file and then read";
        }
    }
    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
    void fillForTest(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, unsigned long offset = 0, bool useOneDigitCoordinates = false)
    {
        unsigned long int coordinateDigits = 100;
        if (useOneDigitCoordinates) {
            coordinateDigits = 10;
        }
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
                        qTmp.q[l] = (((offset * coordinateDigits + i) * coordinateDigits + j) * coordinateDigits + k) * 100 + l;
                    }
                    cu.Q[cu.index(i, j, k)] = qTmp;

                    cu.F[cu.index(i, j, k)].x = (((offset * coordinateDigits + i) * coordinateDigits + j) * coordinateDigits + k) * 100;
                    cu.F[cu.index(i, j, k)].y = (((offset * coordinateDigits + i) * coordinateDigits + j) * coordinateDigits + k) * 100 + 1;
                    cu.F[cu.index(i, j, k)].z = (((offset * coordinateDigits + i) * coordinateDigits + j) * coordinateDigits + k) * 100 + 2;

                    cu.Nu[cu.index(i, j, k)] = (((offset * coordinateDigits + i) * coordinateDigits + j) * coordinateDigits + k) * 100 + 1;
                    cu.O[cu.index(i, j, k)] = true;
                }
            }
        }
#endif
    }

    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
    void generateTestData(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, std::string headerPath, std::string suffix, 
        bool append = false, unsigned long offset = 0, bool markChanged = false, bool useOneDigitCoordinates = false)
    {
        unsigned long int coordinateDigits = 100;
        if (useOneDigitCoordinates) {
            coordinateDigits = 10;
        }
        std::ios_base::openmode mode = std::ios_base::out;
        if (append) {
            mode |= std::ios_base::app;
        }
        std::ofstream hdr(headerPath, mode);
        hdr << "namespace TestUtils {\n";
        hdr << "    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>\n";
        hdr << "    void fillExpectedComputeUnitValues" << suffix << "(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu) {\n";
        hdr << "        QVec<T, QVecSize> qTmp;\n";
        std::string ind47(47, ' ');
        std::string ind19(19, ' ');
        std::string ind36(36, ' ');
        std::string ind23(23, ' ');
        std::string ind13(13, ' ');
        std::string ind8(8, ' ');
        auto m =[&](T v, tNi i, tNi j, tNi k, int l) {
            std::stringstream ss;
            ss << std::setw(7) << v << "ul";
            if ((markChanged) && ((((offset * coordinateDigits + i) * coordinateDigits + j) * coordinateDigits + k)*100+l != v)) {
                ss << "/*C*/";
            }
            return ss.str();
        };
        for (tNi i = 0; i < cu.xg; i++)
        {
            for (tNi j = 0; j < cu.yg; j++)
            {
                for (tNi k = 0; k < cu.zg; k++)
                {
                    hdr << ind47 << " qTmp.q[Q11] = " << m(cu.Q[cu.index(i, j, k)].q[Q11], i, j, k, Q11) << ";\n";
                    hdr << ind19 << " qTmp.q[Q14] = " << m(cu.Q[cu.index(i, j, k)].q[Q14], i, j, k, Q14) << ";";
                    hdr << " qTmp.q[Q03] = " << m(cu.Q[cu.index(i, j, k)].q[Q03], i, j, k, Q03) << ";";
                    hdr << " qTmp.q[Q07] = " << m(cu.Q[cu.index(i, j, k)].q[Q07], i, j, k, Q07) << ";\n";
                    hdr << ind36 << " qTmp.q[Q17] = " << m(cu.Q[cu.index(i, j, k)].q[Q17], i, j, k, Q17) << ";\n";

                    hdr << ind23 << " qTmp.q[Q16] = " << m(cu.Q[cu.index(i, j, k)].q[Q16], i, j, k, Q16) << ";";
                    hdr << " qTmp.q[Q05] = " << m(cu.Q[cu.index(i, j, k)].q[Q05], i, j, k, Q05) << ";";
                    hdr << " qTmp.q[Q09] = " << m(cu.Q[cu.index(i, j, k)].q[Q09], i, j, k, Q09) << ";\n";
                    hdr << ind19 << " qTmp.q[Q02] = " << m(cu.Q[cu.index(i, j, k)].q[Q02], i, j, k, Q02) << ";" << ind23;
                    hdr << " qTmp.q[Q01] = " << m(cu.Q[cu.index(i, j, k)].q[Q01], i, j, k, Q01) << ";\n";
                    hdr << ind13 << " qTmp.q[Q10] = " << m(cu.Q[cu.index(i, j, k)].q[Q10], i, j, k, Q10) << ";";
                    hdr << " qTmp.q[Q06] = " << m(cu.Q[cu.index(i, j, k)].q[Q06], i, j, k, Q06) << ";";
                    hdr << " qTmp.q[Q15] = " << m(cu.Q[cu.index(i, j, k)].q[Q15], i, j, k, Q15) << ";\n";

                    hdr << ind47 << " qTmp.q[Q18] = " << m(cu.Q[cu.index(i, j, k)].q[Q18], i, j, k, Q18) << ";\n";
                    hdr << ind19 << " qTmp.q[Q08] = " << m(cu.Q[cu.index(i, j, k)].q[Q08], i, j, k, Q08) << ";";
                    hdr << " qTmp.q[Q04] = " << m(cu.Q[cu.index(i, j, k)].q[Q04], i, j, k, Q04) << ";";
                    hdr << " qTmp.q[Q13] = " << m(cu.Q[cu.index(i, j, k)].q[Q13], i, j, k, Q13) << ";\n";
                    hdr << ind36 << " qTmp.q[Q12] = " << m(cu.Q[cu.index(i, j, k)].q[Q12], i, j, k, Q12) << ";\n";

                    hdr << "\n        cu.Q[cu.index(" << i << ", " << j << ", " << k << ")] = qTmp;\n";
                    hdr << ind8 << "cu.F[cu.index(" << i << ", " << j << ", " << k << ")].x = " << cu.F[cu.index(i, j, k)].x << ";\n";
                    hdr << ind8 << "cu.F[cu.index(" << i << ", " << j << ", " << k << ")].y = " << cu.F[cu.index(i, j, k)].y << ";\n";
                    hdr << ind8 << "cu.F[cu.index(" << i << ", " << j << ", " << k << ")].z = " << cu.F[cu.index(i, j, k)].z << ";\n";
                }
            }
        }
        hdr << "    }\n}\n";
        hdr.close();
    }
}
