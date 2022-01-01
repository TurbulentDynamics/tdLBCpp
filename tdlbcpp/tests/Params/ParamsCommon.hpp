#include <random>

#include "Params/CheckpointParams.hpp"
#include "Params/BinFileParams.hpp"
#include "Params/ComputeUnitParams.hpp"
#include "Params/FlowParams.hpp"
#include "Params/GridParams.hpp"
#include "Params/RunningParams.hpp"
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
    #if WITH_GPU == 1
    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
    void __global__ initTestData(ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType> &cu, unsigned long offset = 0) {
        tNi i = blockIdx.x * blockDim.x + threadIdx.x;
        tNi j = blockIdx.y * blockDim.y + threadIdx.y;
        tNi k = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (i >= cu.xg || j >= cu.yg || k >= cu.zg) {
            return;
        }

        QVec<T, QVecSize> qTmp;

        for (unsigned long int l = 0; l < QVecSize; l++)
        {
            qTmp.q[l] = offset * 100000000ul + i * 1000000 + j * 10000 + k * 100 + l;
        }
        cu.Q[cu.index(i, j, k)] = qTmp;

        cu.F[cu.index(i, j, k)].x = offset * 100000000ul + i * 1000000 + j * 10000 + k * 100;
        cu.F[cu.index(i, j, k)].y = offset * 100000000ul + i * 1000000 + j * 10000 + k * 100 + 1;
        cu.F[cu.index(i, j, k)].z = offset * 100000000ul + i * 1000000 + j * 10000 + k * 100 + 2;

        cu.Nu[cu.index(i, j, k)] = offset * 100000000ul + i * 1000000 + j * 10000 + k * 100 + 1;
        cu.O[cu.index(i, j, k)] = true;
    }
    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
    void fillForTestGpu(ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType> &cu, unsigned long offset = 0) {

        initTestData<<<cu.numBlocks, cu.threadsPerBlock>>>(*cu.gpuThis, offset);        
    }
    #endif

    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
    void fillForTest(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, unsigned long offset = 0, bool useOneDigitCoordinates = false)
    {
        unsigned long int coordinateDigits = 100;
        if (useOneDigitCoordinates)
        {
            coordinateDigits = 10;
        }

        if (cu.xg > 99 || cu.yg > 99 || cu.zg > 99)
        {
            std::cout << "Size too large for testing" << std::endl;
            exit(1);
        }

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
    }

    void generateTestDataHeader(std::ostream &str, std::string fname);

    void generateTestDataFooter(std::ostream &str);

    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
    void generateTestDataForComputeUnit(std::ostream &str, ComputeUnitBase<T, QVecSize, MemoryLayout> &cu,
                                        unsigned long offset, bool markChanged, bool useOneDigitCoordinates, std::string cuName)
    {
        LOG("generateTestDataForComputeUnit: offset = %lu, markChanged = %d, useOneDigitCoordinates = %d\n", offset, markChanged, useOneDigitCoordinates);
        unsigned long int coordinateDigits = 100;
        if (useOneDigitCoordinates)
        {
            coordinateDigits = 10;
        }

        std::string ind8(7, ' ');
        auto changed = [&](tNi i, tNi j, tNi k, int l)
        {
            return ((((offset * coordinateDigits + i) * coordinateDigits + j) * coordinateDigits + k) * 100 + l != cu.Q[cu.index(i, j, k)].q[l]);
        };
        auto m = [&](T v, tNi i, tNi j, tNi k, int l)
        {
            std::stringstream ss; 
            ss << std::setw(7) << v << "ul";
            if ((markChanged) && changed(i, j, k, l))
            {
                ss << "/*C*/";
            }
            return ss.str();
        };
        auto changedForce = [&](T f, tNi i, tNi j, tNi k, int l)
        {
            return (((offset * coordinateDigits + i) * coordinateDigits + j) * coordinateDigits + k) * 100 + l != f;
        };
        for (tNi i = 0; i < cu.xg; i++)
        {
            for (tNi j = 0; j < cu.yg; j++)
            {
                for (tNi k = 0; k < cu.zg; k++)
                {
                    std::stringstream currentQupdate;
                    for (int l = 0; l < D3Q19; l++)
                    {

                        if (changed(i, j, k, l))
                        {
                            currentQupdate << ind8 << " qTmp.q[Q" << std::setfill('0') << std::setw(2) << (l + 1) << "] = " 
                                           << std::setw(0) << std::setfill(' ') << m(cu.Q[cu.index(i, j, k)].q[l], i, j, k, l) << ";\n";
                        }
                    }

                    if (currentQupdate.rdbuf()->in_avail() > 0)
                    {
                        str << ind8 << " qTmp = " << cuName << "Q[" << cuName << "index(" << i << ", " << j << ", " << k << ")];\n";
                        str << currentQupdate.rdbuf();
                        str << "        " << cuName << "Q[" << cuName << "index(" << i << ", " << j << ", " << k << ")] = qTmp;\n\n";
                    }
                    if (changedForce(cu.F[cu.index(i, j, k)].x, i, j, k, 0))
                    {
                        str << ind8 << cuName << "F[" << cuName << "index(" << i << ", " << j << ", " << k << ")].x = " << cu.F[cu.index(i, j, k)].x << ";\n";
                    }
                    if (changedForce(cu.F[cu.index(i, j, k)].y, i, j, k, 1))
                    {
                        str << ind8 << cuName << "F[" << cuName << "index(" << i << ", " << j << ", " << k << ")].y = " << cu.F[cu.index(i, j, k)].y << ";\n";
                    }
                    if (changedForce(cu.F[cu.index(i, j, k)].z, i, j, k, 2))
                    {
                        str << ind8 << cuName << "F[" << cuName << "index(" << i << ", " << j << ", " << k << ")].z = " << cu.F[cu.index(i, j, k)].z << ";\n";
                    }
                }
            }
        }
    }

    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
    void generateTestData(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, std::string headerPath, std::string suffix,
                          bool append = false, unsigned long offset = 0, bool markChanged = false, bool useOneDigitCoordinates = false)
    {
        std::ios_base::openmode mode = std::ios_base::out;
        if (append)
        {
            mode |= std::ios_base::app;
        }
        std::ofstream hdr(headerPath, mode);
        generateTestDataHeader(hdr, std::string("fillExpectedComputeUnitValues") + suffix);
        generateTestDataForComputeUnit(hdr, cu, offset, markChanged, useOneDigitCoordinates, std::string("cu."));
        generateTestDataFooter(hdr);

        hdr.close();
    }

#if WITH_MPI == 1
    void generateTestDataHeaderMpi(std::ostream &str, std::string fname);

    template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, MemoryLayoutType MemoryLayoutHalo>
    void generateTestDataMpi(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, ComputeUnitBase<T, QVecSize, MemoryLayoutHalo> **halos,
                             int rank, int numprocs, MPI_Comm comm, std::string headerPath, std::string suffix,
                             bool markChanged = false, bool useOneDigitCoordinates = false)
    {
        std::ofstream *hdr;
        std::ios_base::openmode mode = std::ios_base::out;
        if (rank != 0)
        {
            mode |= std::ios_base::app;
        }
        else
        {
            hdr = new std::ofstream(headerPath, mode);
        }
        MPI_Barrier(comm);
        for (int i = 0; i < numprocs; i++)
        {
            if (rank == i)
            {
                if (i != 0)
                {
                    hdr = new std::ofstream(headerPath, mode);
                }
                if (i == 0)
                {
                    generateTestDataHeaderMpi(*hdr, std::string("fillExpectedComputeUnitValues") + suffix);
                }
                (*hdr) << "    if (Mpi.rank == " << rank << ") {\n";
                generateTestDataForComputeUnit(*hdr, cu, rank * 100, markChanged, useOneDigitCoordinates, std::string("cu."));
                for (int j = 0; j < D3Q27; j++)
                {
                    generateTestDataForComputeUnit(*hdr, *halos[j], rank * 100 + j + 1, markChanged, useOneDigitCoordinates,
                                                   std::string("halos[") + std::to_string(j) + std::string("]->"));
                }
                (*hdr) << "    }\n";
                if (i == numprocs - 1)
                {
                    generateTestDataFooter(*hdr);
                }
                hdr->close();
            }
            MPI_Barrier(comm);
        }
    }
#endif
}
