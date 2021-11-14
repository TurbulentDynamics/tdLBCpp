#pragma once

#include "ComputeUnit.h"
#include "ComputeUnitGpu.h"
#include "Forcing.cuh"

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
    F = new Force<T>[size];
    checkCudaErrors(cudaMalloc((void **)&devF, sizeof(Force<T>) * size));
    checkCudaErrors(cudaMalloc((void **)&Nu, sizeof(T) * size));
    checkCudaErrors(cudaMalloc((void **)&O, sizeof(T) * size));
    ExcludeOutputPoints = new bool[size];
    checkCudaErrors(cudaMalloc((void **)&gpuThis, sizeof(ComputeUnitBase<T, QVecSize, MemoryLayout>)));
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
        delete[] F;
        F = nullptr;
    }
    if (devF != nullptr)
    {
        checkCudaErrors(cudaFree(devF));
        devF = nullptr;
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

    F = new Force<T>[size];

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

    size_t objectSize = sizeof(ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>);
    if (gpuThis == nullptr)
    {
        checkCudaErrors(cudaMalloc((void **)&gpuThis, objectSize));
    }
    checkCudaErrors(cudaMemcpy(gpuThis, this, objectSize, cudaMemcpyHostToDevice));
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::ComputeUnitArchitecture(ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree) : 
    ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>(cuParams, flow, outputTree)
{
    gpuGeomSize = 0;
    gpuGeom = nullptr;
    init(cuParams, false);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::ComputeUnitArchitecture(ComputeUnitArchitecture &&rhs) noexcept : 
    ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>(rhs),
    devF(rhs.devF), threadsPerBlock(rhs.threadsPerBlock), numBlocks(rhs.numBlocks), gpuThis(rhs.gpuThis)
{
    rhs.devF = nullptr;
    rhs.gpuThis = nullptr;
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::initialise(T initialRho)
{
    checkCudaErrors(cudaMemcpy(gpuThis, this, sizeof(ComputeUnitBase<T, QVecSize, MemoryLayout>), cudaMemcpyHostToDevice));
    ::setQToZero<<<numBlocks, threadsPerBlock>>>(*gpuThis);
    //setRhoTo<<< numBlocks, threadsPerBlock >>>(*this, flow.initialRho);
    //setForceToZero<<< numBlocks, threadsPerBlock >>>(*this, initialRho);
    //if (flow.useLES){
    //setNuToZero<<< numBlocks, threadsPerBlock >>>(*this, initialRho);
    //}
    //setOToZero<<< numBlocks, threadsPerBlock >>>(*this, initialRho);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>::forcing(std::vector<PosPolar<tNi, T>> &geom, T alfa, T beta) {
    ::setOToZero<<<numBlocks, threadsPerBlock>>>(*this);
    int blocks = (geom.size() + threadsPerBlock.x - 1) / threadsPerBlock.x;
    if (gpuGeomSize < geom.size()) {
        if (gpuGeom != nullptr) {
            checkCudaErrors(cudaFree(gpuGeom));
        }
        gpuGeomSize = geom.size();
        checkCudaErrors(cudaMalloc((void **)&gpuGeom, sizeof(PosPolar<tNi, T>) * gpuGeomSize));
    }
    checkCudaErrors(cudaMemcpy(gpuGeom, &geom[0], sizeof(PosPolar<tNi, T>) *geom.size(), cudaMemcpyHostToDevice));
    ::forcing<T, QVecSize, MemoryLayout, collisionType, streamingType><<<blocks, threadsPerBlock>>>(*this, gpuGeom, geom.size(), alfa, beta);
    ::setFToZeroWhenOIsZero<<<numBlocks, threadsPerBlock>>>(*this);
}