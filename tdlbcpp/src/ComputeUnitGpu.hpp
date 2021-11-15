#pragma once

#include "ComputeUnit.h"
#include "ComputeUnitGpu.h"

#include "Forcing.cuh"
#include "ComputeUnit.cuh"
#include "CollisionEgglesSomers.cuh"
#include "Boundary.cuh"
#include "StreamingNieve.cuh"
#include "StreamingEsoTwist.cuh"
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::checkEnoughMemory()
{
    checkCudaErrors(cudaSetDevice(deviceID));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceID);

    unsigned long long memRequired = size * (sizeof(T) * QVecSize + sizeof(Force<T>) + sizeof(T) + sizeof(bool) + sizeof(bool));

    if (memRequired > prop.totalGlobalMem)
    {
        std::cout << "Cannot allocate device on GPU." << std::endl;
        exit(1);
    }
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::allocateMemory()
{
    checkEnoughMemory();

    Q.setSize(size);
    checkCudaErrors(cudaMalloc((void **)&Q.q, sizeof(T) * Q.qSize));
    checkCudaErrors(cudaMalloc((void **)&F, sizeof(Force<T>) * size));
    checkCudaErrors(cudaMalloc((void **)&Nu, sizeof(T) * size));
    checkCudaErrors(cudaMalloc((void **)&O, sizeof(T) * size));
    ExcludeOutputPoints = new bool[size];
    checkCudaErrors(cudaMalloc((void **)&gpuThis, sizeof(Current)));
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::freeMemory()
{
    if (gpuThis != nullptr)
    {
        checkCudaErrors(cudaFree(gpuThis));
        gpuThis = nullptr;
    }
    if (Q.q != nullptr)
    {
        checkCudaErrors(cudaFree(Q.q));
        Q.q = nullptr;
    }
    if (F != nullptr)
    {
        checkCudaErrors(cudaFree(F));
        F = nullptr;
    }
    if (Nu != nullptr)
    {
        checkCudaErrors(cudaFree(Nu));
        Nu = nullptr;
    }
    if (O != nullptr)
    {
        checkCudaErrors(cudaFree(O));
        O = nullptr;
    }
    if (ExcludeOutputPoints != nullptr)
    {
        delete[] ExcludeOutputPoints;
        ExcludeOutputPoints = nullptr;
    }
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU_MEM_SHARED>::allocateMemory()
{
    checkEnoughMemory();

    Q.setSize(size);
    checkCudaErrors(cudaMalloc((void **)&Q.q, sizeof(T) * Q.qSize));

    //F = new Force<T>[size];

    checkCudaErrors(cudaHostAlloc((void **)&devF, sizeof(T) * size, cudaHostAllocMapped));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&devF, (void *)F, 0));

    checkCudaErrors(cudaHostAlloc((void **)&O, sizeof(T) * size, cudaHostAllocMapped));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&O, (void *)F, 0));

    checkCudaErrors(cudaHostAlloc((void **)&Nu, sizeof(T) * size, cudaHostAllocMapped));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&Nu, (void *)F, 0));

    ExcludeOutputPoints = new bool[size];
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU_MEM_SHARED>::freeMemory()
{
    if (Q.q != nullptr)
    {
        checkCudaErrors(cudaFree(Q.q));
        Q.q = nullptr;
    }
    if (gpuThis != nullptr)
    {
        checkCudaErrors(cudaFree(gpuThis));
        gpuThis = nullptr;
    }
    if (F != nullptr)
    {
        checkCudaErrors(cudaFree(F));
        F = nullptr;
    }
    if (Nu != nullptr)
    {
        checkCudaErrors(cudaFree(Nu));
        Nu = nullptr;
    }
    if (O != nullptr)
    {
        checkCudaErrors(cudaFree(O));
        O = nullptr;
    }
    if (ExcludeOutputPoints != nullptr)
    {
        delete[] ExcludeOutputPoints;
        ExcludeOutputPoints = nullptr;
    }
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::architectureInit()
{

    int threads_per_warp = 32;
    int max_threads_per_block = 512;

    int xthreads_per_block = 8;
    int ythreads_per_block = 8;
    int zthreads_per_block = 8;

    threadsPerBlock = dim3(xthreads_per_block, ythreads_per_block, zthreads_per_block);

    int block_in_x_dirn = xg / threadsPerBlock.x + (xg % xthreads_per_block != 0);
    int block_in_y_dirn = zg / threadsPerBlock.y + (yg % ythreads_per_block != 0);
    int block_in_z_dirn = zg / threadsPerBlock.z + (zg % zthreads_per_block != 0);

    numBlocks = dim3(block_in_x_dirn, block_in_y_dirn, block_in_z_dirn);

    std::cout << "threads_per_block" << threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z << std::endl;
    std::cout << "numBlocks" << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << std::endl;
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::ComputeUnitArchitecture(ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree) : 
    ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>(cuParams, flow, outputTree)
{
    gpuThis = nullptr;
    std::cout << "gpuThis = " << gpuThis << std::endl;
    gpuGeomSize = 0;
    gpuGeom = nullptr;
    init(cuParams, false);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::ComputeUnitArchitecture(ComputeUnitArchitecture &&rhs) noexcept : 
    ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>(rhs),
    threadsPerBlock(rhs.threadsPerBlock), numBlocks(rhs.numBlocks), gpuThis(rhs.gpuThis)
{
    rhs.gpuThis = nullptr;
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::initialise(T initialRho)
{
    checkCudaErrors(cudaMemcpy(gpuThis, this, sizeof(Current), cudaMemcpyHostToDevice));
    ::setQToZero<<<numBlocks, threadsPerBlock>>>(*gpuThis);
    ::setRhoTo<<< numBlocks, threadsPerBlock >>>(*gpuThis, flow.initialRho);
    ::setForceToZero<<< numBlocks, threadsPerBlock >>>(*gpuThis);
    if (flow.useLES){
        ::setNuToZero<<< numBlocks, threadsPerBlock >>>(*gpuThis, initialRho);
    }
    ::setOToZero<<< numBlocks, threadsPerBlock >>>(*gpuThis);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::forcing(std::vector<PosPolar<tNi, T>> &geom, T alfa, T beta) {
    ::setOToZero<<<numBlocks, threadsPerBlock>>>(*gpuThis);
    int blocks = (geom.size() + threadsPerBlock.x - 1) / threadsPerBlock.x;
    if (gpuGeomSize < geom.size()) {
        if (gpuGeom != nullptr) {
            checkCudaErrors(cudaFree(gpuGeom));
        }
        gpuGeomSize = geom.size();
        checkCudaErrors(cudaMalloc((void **)&gpuGeom, sizeof(PosPolar<tNi, T>) * gpuGeomSize));
    }
    checkCudaErrors(cudaMemcpy(gpuGeom, &geom[0], sizeof(PosPolar<tNi, T>) *geom.size(), cudaMemcpyHostToDevice));
    ::forcing<T, QVecSize, MemoryLayout, collisionType, streamingType><<<blocks, threadsPerBlock>>>(*gpuThis, gpuGeom, geom.size(), alfa, beta);
    ::setFToZeroWhenOIsZero<<<numBlocks, threadsPerBlock>>>(*gpuThis);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::collision() {
    ::collision<<<numBlocks, threadsPerBlock>>>(*gpuThis);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::moments() {
    ::moments<<<numBlocks, threadsPerBlock>>>(*gpuThis);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::bounceBackBoundary() {
    dim3 blocksXY(numBlocks.x, numBlocks.y);
    dim3 threadsXY(threadsPerBlock.x, threadsPerBlock.y);
    ::bounceBackBoundaryBackward<<<blocksXY, threadsXY>>>(*gpuThis);
    ::bounceBackBoundaryForward<<<blocksXY, threadsXY>>>(*gpuThis);
    dim3 blocksXZ(numBlocks.x, numBlocks.z);
    dim3 threadsXZ(threadsPerBlock.x, threadsPerBlock.z);
    ::bounceBackBoundaryUp<<<blocksXZ, threadsXZ>>>(*gpuThis);
    ::bounceBackBoundaryDown<<<blocksXZ, threadsXZ>>>(*gpuThis);
    dim3 blocksYZ(numBlocks.y, numBlocks.z);
    dim3 threadsYZ(threadsPerBlock.y, threadsPerBlock.z);
    ::bounceBackBoundaryRight<<<blocksYZ, threadsYZ>>>(*gpuThis);
    ::bounceBackBoundaryLeft<<<blocksYZ, threadsYZ>>>(*gpuThis);

    ::bounceBackEdges<<<numBlocks.z, threadsPerBlock.z>>>(*gpuThis, Q05);
    ::bounceBackEdges<<<numBlocks.y, threadsPerBlock.y>>>(*gpuThis, Q03);
    ::bounceBackEdges<<<numBlocks.x, threadsPerBlock.x>>>(*gpuThis, Q01);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::streamingPush() {
    ::streamingPush<<<1, 2>>>(*gpuThis);
}