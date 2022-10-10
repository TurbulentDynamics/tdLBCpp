#pragma once

#include "ComputeUnit.h"
#include "ComputeUnitGpu.h"

#include "Forcing.cuh"
#include "ComputeUnit.cuh"
#include "CollisionEgglesSomers.cuh"
#include "Boundary.cuh"
#include "StreamingSimple.cuh"
#include "StreamingEsoTwist.cuh"
#include "ComputeUnitOutput.cuh"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::checkEnoughMemory()
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
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::allocateMemory()
{
    checkEnoughMemory();

    Q.setSize(size);
    checkCudaErrors(cudaMalloc((void **)&Q.q, sizeof(T) * Q.qSize));
    checkCudaErrors(cudaMalloc((void **)&F, sizeof(Force<T>) * size));
    checkCudaErrors(cudaMalloc((void **)&Nu, sizeof(T) * size));
    checkCudaErrors(cudaMalloc((void **)&O, sizeof(T) * size));
    ExcludeOutputPoints = new bool[size];
    checkCudaErrors(cudaMalloc((void **)&devExcludeOutputPoints, sizeof(bool) * size));
    checkCudaErrors(cudaMalloc((void **)&gpuThis, sizeof(Current)));
    checkCudaErrors(cudaMalloc((void **)&VortXY, sizeof(T) * size_t(xg) * yg));
    checkCudaErrors(cudaMalloc((void **)&VortXZ, sizeof(T) * size_t(xg) * zg));
    checkCudaErrors(cudaMalloc((void **)&VortYZ, sizeof(T) * size_t(yg) * zg));
    std::cout << "GPU Memory allocated" << std::endl;
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::freeMemory()
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
    if (VortXY != nullptr)
    {
        checkCudaErrors(cudaFree(VortXY));
        VortXY = nullptr;
    }
    if (VortXZ != nullptr)
    {
        checkCudaErrors(cudaFree(VortXZ));
        VortXZ = nullptr;
    }
    if (VortYZ != nullptr)
    {
        checkCudaErrors(cudaFree(VortYZ));
        VortYZ = nullptr;
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
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::architectureInit()
{

    //int threads_per_warp = 32;
    //int max_threads_per_block = 512;

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
ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::ComputeUnitArchitectureCommonGPU(ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree, bool allocate) :
    gpuThis(nullptr), gpuGeomSize(0), gpuGeom(nullptr), VortXY(nullptr), VortXZ(nullptr), VortYZ(nullptr),
    ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>(cuParams, flow, outputTree, false)
{
    size = 0;
    init(cuParams, true);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::ComputeUnitArchitectureCommonGPU(ComputeUnitArchitectureCommonGPU &&rhs) noexcept : 
    ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>(rhs),
    threadsPerBlock(rhs.threadsPerBlock), numBlocks(rhs.numBlocks), gpuThis(rhs.gpuThis), VortXY(rhs.VortXY), VortXZ(rhs.VortXZ), VortYZ(rhs.VortYZ),
    gpuGeomSize(rhs.gpuGeomSize), gpuGeom(rhs.gpuGeom)
{
    rhs.gpuThis = nullptr;
    rhs.gpuGeomSize = 0;
    rhs.gpuGeom = nullptr;
    rhs.VortXY = nullptr;
    rhs.VortXZ = nullptr;
    rhs.VortYZ = nullptr;
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::~ComputeUnitArchitectureCommonGPU()
{
    freeMemory();
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::initialise(T initialRho)
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
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
int ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::containsErrors()
{
    checkCudaErrors(cudaMemcpy(gpuThis, this, sizeof(Current), cudaMemcpyHostToDevice));
    ::containsErrorsInQ<<<numBlocks, threadsPerBlock>>>(*gpuThis);
    
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::doubleResolutionFullCU()
{
    // save old fields
    Fld oldQ;
    oldQ.setSize(size);
    oldQ.q = new T[oldQ.qSize];
    checkCudaErrors(cudaMemcpy(oldQ.q, Q.q, sizeof(T) * oldQ.qSize, cudaMemcpyDeviceToHost));

    T *oldNu;
    oldNu = new T[size];
    checkCudaErrors(cudaMemcpy(oldNu, Nu, sizeof(T) * size, cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // set new grid size
    tNi factor = 2;

    resolution += factor;

    x *= factor;
    y *= factor;
    z *= factor;

    initGridParams();

    size = size_t(xg) * yg * zg;

    freeMemory();
    allocateMemory();//using new cu.x, size etc
    architectureInit();
    initialise(0);

    Fld newQ;
    newQ.setSize(size);
    newQ.q = new T[newQ.qSize];
    T *newNu;
    newNu = new T[size];
    
    copyFieldsWithScaling(newQ, oldQ, newNu, oldNu);
    
    checkCudaErrors(cudaMemcpy(Q.q, newQ.q, sizeof(T) * newQ.qSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(Nu, newNu, sizeof(T) * size, cudaMemcpyHostToDevice));

    delete[] oldQ.q;
    delete[] oldNu;
    delete[] newQ.q;
    delete[] newNu;

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::forcing(std::vector<PosPolar<tNi, T>> &geom, T alfa, T beta) {
    ::setOToZero<<<numBlocks, threadsPerBlock>>>(*gpuThis);
    int blocks = (geom.size() + threadsPerBlock.x - 1) / threadsPerBlock.x;
    if (gpuGeomSize < geom.size()) {
        if (gpuGeom != nullptr) {
            checkCudaErrors(cudaFree(gpuGeom));
        }
        gpuGeomSize = geom.size();
        size_t gpuGeomSizeBytes = sizeof(PosPolar<tNi, T>) * gpuGeomSize;
        LOG("gpuGeom resize, new size: %ld\n", gpuGeomSizeBytes);
        checkCudaErrors(cudaMalloc((void **)&gpuGeom, gpuGeomSizeBytes));
    }
    checkCudaErrors(cudaMemcpy(gpuGeom, &geom[0], sizeof(PosPolar<tNi, T>) *geom.size(), cudaMemcpyHostToDevice));
    ::forcing<T, QVecSize, MemoryLayout, collisionType, streamingType><<<blocks, threadsPerBlock.x>>>(*gpuThis, gpuGeom, geom.size(), alfa, beta);
    ::setFToZeroWhenOIsZero<<<numBlocks, threadsPerBlock>>>(*gpuThis);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::checkpoint_read(std::string dirname, std::string unit_name) {
    ComputeUnitBase<T, QVecSize, MemoryLayout>::checkpoint_read(dirname, unit_name);
    checkCudaErrors(cudaMemcpy(gpuThis, this, sizeof(Current), cudaMemcpyHostToDevice));
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
void ComputeUnitArchitectureCollision<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>::collision() {
    ::collisionEgglesSommers<<<numBlocks, threadsPerBlock>>>(*gpuThis);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
void ComputeUnitArchitectureCollision<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>::moments() {
    ::moments<<<numBlocks, threadsPerBlock>>>(*gpuThis);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, Simple, GPU>::bounceBackBoundary() {
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

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, Simple, GPU>::streamingPush() {
    dim3 blocksYZ(numBlocks.y, numBlocks.z, 1);
    dim3 threadsYZ(threadsPerBlock.y, threadsPerBlock.z, 2);
    ::streamingPushDir<<<blocksYZ, threadsYZ>>>(*gpuThis, Q01);
    dim3 blocksXZ(numBlocks.x, numBlocks.z, 1);
    dim3 threadsXZ(threadsPerBlock.x, threadsPerBlock.z, 2);
    ::streamingPushDir<<<blocksXZ, threadsXZ>>>(*gpuThis, Q03);
    dim3 blocksXY(numBlocks.x, numBlocks.y, 1);
    dim3 threadsXY(threadsPerBlock.x, threadsPerBlock.y, 2);
    ::streamingPushDir<<<blocksXY, threadsXY>>>(*gpuThis, Q05);
    dim3 blocksZXY(numBlocks.z, numBlocks.x + numBlocks.y, 1);
    dim3 threadsZXY(threadsPerBlock.z, threadsPerBlock.x, 2);
    ::streamingPushDir<<<blocksZXY, threadsZXY>>>(*gpuThis, Q07);
    dim3 blocksYXZ(numBlocks.y, numBlocks.x + numBlocks.z, 1);
    dim3 threadsYXZ(threadsPerBlock.y, threadsPerBlock.x, 2);
    ::streamingPushDir<<<blocksYXZ, threadsYXZ>>>(*gpuThis, Q09);
    dim3 blocksXYZ(numBlocks.x, numBlocks.y + numBlocks.z, 1);
    dim3 threadsXYZ(threadsPerBlock.x, threadsPerBlock.y, 2);
    ::streamingPushDir<<<blocksXYZ, threadsXYZ>>>(*gpuThis, Q11);

    ::streamingPushDir<<<blocksZXY, threadsZXY>>>(*gpuThis, Q13);
    ::streamingPushDir<<<blocksYXZ, threadsYXZ>>>(*gpuThis, Q15);
    ::streamingPushDir<<<blocksXYZ, threadsXYZ>>>(*gpuThis, Q17);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, Esotwist, GPU>::streamingPush() {
    evenStep = !evenStep;
    checkCudaErrors(cudaMemcpy(gpuThis, this, sizeof(Current), cudaMemcpyHostToDevice));
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::calcVorticityXY(tNi k, RunningParams runParam, int jpegCompression) {
    T *Vort = new(T[xg * yg]);
    dim3 blocksXY(numBlocks.x, numBlocks.y, 1);
    dim3 threadsXY(threadsPerBlock.x, threadsPerBlock.y, 1);
    ::calcVorticityXY<<<blocksXY, threadsXY>>>(*gpuThis, k, jpegCompression);
    checkCudaErrors(cudaMemcpy(Vort, VortXY, sizeof(T) * xg * yg, cudaMemcpyDeviceToHost));
    saveJpeg("xy", Vort, xg, yg, 1, runParam, k);
    delete []Vort;
}
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::calcVorticityXZ(tNi j, RunningParams runParam, int jpegCompression) {
    T *Vort = new(T[xg * zg]);
    dim3 blocksXZ(numBlocks.x, 1, numBlocks.z);
    dim3 threadsXZ(threadsPerBlock.x, 1, threadsPerBlock.z);
    ::calcVorticityXZ<<<blocksXZ, threadsXZ>>>(*gpuThis, j, jpegCompression);
    checkCudaErrors(cudaMemcpy(Vort, VortXZ, sizeof(T) * xg * zg, cudaMemcpyDeviceToHost));
    saveJpeg("xz", Vort, xg, zg, 1, runParam, j);
    delete []Vort;
}
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::calcVorticityYZ(tNi i, RunningParams runParam, int jpegCompression) {
    T *Vort = new(T[yg * zg]);
    dim3 blocksYZ(numBlocks.y, numBlocks.z, 1);
    dim3 threadsYZ(threadsPerBlock.y, threadsPerBlock.z, 1);
    ::calcVorticityYZ<<<blocksYZ, threadsYZ>>>(*gpuThis, i, jpegCompression);
    checkCudaErrors(cudaMemcpy(Vort, VortYZ, sizeof(T) * yg * zg, cudaMemcpyDeviceToHost));
    saveJpeg("yz", Vort, yg, zg, 1, runParam, i);
    delete []Vort;
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::setOutputExcludePoints(std::vector<Pos3d<tNi>> geomPoints){
    ComputeUnitBase<T, QVecSize, MemoryLayout>::setOutputExcludePoints(geomPoints);
    checkCudaErrors(cudaMemcpy(devExcludeOutputPoints, ExcludeOutputPoints, sizeof(bool) * size, cudaMemcpyHostToDevice));
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::setOutputExcludePoints(std::vector<PosPolar<tNi, T>> geomPoints){
    ComputeUnitBase<T, QVecSize, MemoryLayout>::setOutputExcludePoints(geomPoints);
    checkCudaErrors(cudaMemcpy(devExcludeOutputPoints, ExcludeOutputPoints, sizeof(bool) * size, cudaMemcpyHostToDevice));
}



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::unsetOutputExcludePoints(std::vector<Pos3d<tNi>> geomPoints){
    ComputeUnitBase<T, QVecSize, MemoryLayout>::unsetOutputExcludePoints(geomPoints);
    checkCudaErrors(cudaMemcpy(devExcludeOutputPoints, ExcludeOutputPoints, sizeof(bool) * size, cudaMemcpyHostToDevice));
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>::unsetOutputExcludePoints(std::vector<PosPolar<tNi, T>> geomPoints){
    ComputeUnitBase<T, QVecSize, MemoryLayout>::unsetOutputExcludePoints(geomPoints);
    checkCudaErrors(cudaMemcpy(devExcludeOutputPoints, ExcludeOutputPoints, sizeof(bool) * size, cudaMemcpyHostToDevice));
}
