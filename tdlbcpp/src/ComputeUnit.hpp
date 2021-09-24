//
//  ComputeUnit.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include <iostream>
#include <cerrno>

#if WITH_GPU
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

#include "Tools/toojpeg.h"

#include "ComputeUnit.h"
#include "DiskOutputTree.h"


//
//template <typename T, int QVecSize>
//ComputeUnit<T, QVecSize>::ComputeUnit(ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree):flow(flow), outputTree(outputTree){
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::init(ComputeUnitParams cuParams, bool allocateMemory) {

    nodeID = cuParams.nodeID;
    deviceID = cuParams.deviceID;

    idi = cuParams.idi;
    idj = cuParams.idj;
    idk = cuParams.idk;

    x = cuParams.x;
    y = cuParams.y;
    z = cuParams.z;

    i0 = cuParams.i0;
    j0 = cuParams.j0;
    k0 = cuParams.k0;

    ghost = cuParams.ghost;


    xg = x + 2 * ghost;
    yg = y + 2 * ghost;
    zg = z + 2 * ghost;

    //Allows for (tNi i=0; i<=xg0; i++){
    xg0 = xg - 1;
    yg0 = yg - 1;
    zg0 = zg - 1;

    //Allows for (tNi i=1; i<=xg1; i++){
    xg1 = xg - 2;
    yg1 = yg - 2;
    zg1 = zg - 2;


    size_t new_size = size_t(xg) * yg * zg;
    bool sizeChanged = (new_size != size);

    if (allocateMemory && sizeChanged) {
        size = new_size;

        Q.allocate(size);

#ifdef WITH_CPU
        delete[] F;
        delete[] Nu;
        delete[] O;
        delete[] ExcludeOutputPoints;
#endif
#if WITH_GPU
        delete[] F;
        checkCudaErrors(cudaFree(devF));
        checkCudaErrors(cudaFree(Nu));
        checkCudaErrors(cudaFree(O));
        delete[] ExcludeOutputPoints;
#endif
#if WITH_GPU_MEMSHARED == 1
        checkCudaErrors(cudaFree(F));
        checkCudaErrors(cudaFree(Nu));
        checkCudaErrors(cudaFree(O));
        delete[] ExcludeOutputPoints;
#endif
#if defined(WITH_GPU) || defined(WITH_GPU_MEMSHARED)
        checkCudaErrors(cudaSetDevice(deviceID));

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceID);

        unsigned long long memRequired = size * (sizeof(T) * QVecSize + sizeof(Force<T>) + sizeof(T) + sizeof(bool) + sizeof(bool));

        if (memRequired > prop.totalGlobalMem){
            std::cout << "Cannot allocate device on GPU." << std::endl;
            exit(1);
        }
#endif

#ifdef WITH_CPU
        F = new Force<T>[size];
        Nu = new T[size];
        O = new bool[size];
        ExcludeOutputPoints = new bool[size];
#endif

#if WITH_GPU
        F = new Force<T>[size];
        checkCudaErrors(cudaMalloc((void **)&devF, sizeof(Force<T>) * size));
        checkCudaErrors(cudaMalloc((void **)&Nu, sizeof(T) * size));
        checkCudaErrors(cudaMalloc((void **)&O, sizeof(T) * size));
        ExcludeOutputPoints = new bool[size];
#endif

#if WITH_GPU_MEMSHARED == 1
        F = new Force<T>[size];

        checkCudaErrors(cudaHostAlloc((void **)&devF, sizeof(T) * size, cudaHostAllocMapped));
        checkCudaErrors(cudaHostGetDevicePointer((void **)&devF, (void *)F, 0));

        checkCudaErrors(cudaHostAlloc((void **)&O, sizeof(T) * size, cudaHostAllocMapped));
        checkCudaErrors(cudaHostGetDevicePointer((void **)&O, (void *)F, 0));

        checkCudaErrors(cudaHostAlloc((void **)&Nu, sizeof(T) * size, cudaHostAllocMapped));
        checkCudaErrors(cudaHostGetDevicePointer((void **)&Nu, (void *)F, 0));

        ExcludeOutputPoints = new bool[size];
#endif



#if defined(WITH_GPU) || defined(WITH_GPU_MEMSHARED)
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
#endif
    }
    size = new_size;

    evenStep = true;
}





template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
ComputeUnitBase<T, QVecSize, MemoryLayout>::ComputeUnitBase(ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree, bool allocateMemory):
  size(0), flow(flow), outputTree(outputTree) {
    F = nullptr;
    O = nullptr;
    Nu = nullptr;
    ExcludeOutputPoints = nullptr;
    init(cuParams, allocateMemory);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
ComputeUnitBase<T, QVecSize, MemoryLayout>::ComputeUnitBase(ComputeUnitBase &&rhs) noexcept: 
    idi(rhs.idi), idj(rhs.idj), idk(rhs.idk), nodeID(rhs.nodeID), deviceID(rhs.deviceID),
    x(rhs.x), y(rhs.y), z(rhs.z), i0(rhs.i0), j0(rhs.j0), k0(rhs.k0), xg(rhs.xg), yg(rhs.yg), zg(rhs.zg), xg0(rhs.xg0), yg0(rhs.yg0), zg0(rhs.zg0), xg1(rhs.xg1), yg1(rhs.yg1), zg1(rhs.zg1),
    ghost(rhs.ghost), size(rhs.size), flow(rhs.flow), Q(std::move(rhs.Q)), F(rhs.F), Nu(rhs.Nu), O(rhs.O), ExcludeOutputPoints(rhs.ExcludeOutputPoints),
    outputTree(rhs.outputTree), evenStep(rhs.evenStep)
#if defined(WITH_GPU) || defined(WITH_GPU_MEMSHARED)
    , devF(rhs.devF), devN(rhs.devN), devNu(rhs.debNu), threadsPerBlock(rhs.threadsPerBlock), numBlocks(rhs.numBlocks)
#endif
{
    rhs.O = nullptr;
    rhs.Nu = nullptr;
    rhs.F = nullptr;
    rhs.ExcludeOutputPoints = nullptr;
#if WITH_GPU
    rhs.devF = nullptr;
#endif
}



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
ComputeUnitBase<T, QVecSize, MemoryLayout>::~ComputeUnitBase()
{
#if WITH_GPU
    checkCudaErrors(cudaSetDevice(deviceID));
    checkCudaErrors(cudaFree(F));
    checkCudaErrors(cudaFree(devF));
    checkCudaErrors(cudaFree(Nu));
    checkCudaErrors(cudaFree(O));
#endif
#if WITH_GPU_MEMSHARED == 1
        checkCudaErrors(cudaSetDevice(deviceID));
        checkCudaErrors(cudaFree(F));
        checkCudaErrors(cudaFree(Nu));
        checkCudaErrors(cudaFree(O));
#endif
}



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::setQToZero(){

    for (tNi i=0; i<=xg0; i++){
        for (tNi j=0; j<=yg0; j++){
            for (tNi k=0; k<=zg0; k++){

                Q[index(i, j, k)].setToZero();
            }
        }
    }
};




template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::initialise(T initialRho){

#ifdef WITH_CPU
    for (tNi i=0; i<=xg0; i++){
        for (tNi j=0; j<=yg0; j++){
            for (tNi k=0; k<=zg0; k++){

                Q[index(i, j, k)].initialiseRho(initialRho, 0.0);
                F[index(i, j, k)].setToZero();
                Nu[index(i, j, k)] = 0.0;
                O[index(i, j, k)] = false;

                ExcludeOutputPoints[index(i,j,k)] = false;
            }
        }
    }
#endif
#if WITH_GPU || WITH_GPU_SHAREDMEM == 1

    setQToZero<<< numBlocks, threadsPerBlock >>>();
    setRhoTo(flow.initialRho)<<< numBlocks, threadsPerBlock >>>();
    setForceToZero<<< numBlocks, threadsPerBlock >>>();
    if (flow.useLES){
        setNuToZero<<< numBlocks, threadsPerBlock >>>();
    }
    setOToZero<<< numBlocks, threadsPerBlock >>>();

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#endif
};



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
Velocity<T> ComputeUnitBase<T, QVecSize, MemoryLayout>::getVelocity(tNi i, tNi j, tNi k){
    return Q[index(i, j, k)].velocity();
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
Velocity<T> ComputeUnitBase<T, QVecSize, MemoryLayout>::getVelocitySparseF(tNi i, tNi j, tNi k, Force<T> f){
    return Q[index(i, j, k)].velocity(f);
};





template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
FILE* ComputeUnitBase<T, QVecSize, MemoryLayout>::fopen_read(std::string filePath){

    std::cout << "Node " << nodeID << " Load " << filePath << std::endl;

    return fopen(filePath.c_str(), "r");
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
FILE* ComputeUnitBase<T, QVecSize, MemoryLayout>::fopen_write(std::string filePath){

    std::cout << "Node " << nodeID << " Save " << filePath << std::endl;

    return fopen(filePath.c_str(), "w");
}





template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::checkpoint_read(std::string dirname, std::string unit_name){
    
    BinFileParams binFormat;
    binFormat.filePath = outputTree.getAllParamsFilePath(dirname, unit_name);
    binFormat.structName = "checkpoint";
    binFormat.binFileSizeInStructs = (Json::UInt64)size;
    binFormat.coordsType = "none";
    binFormat.hasGridtCoords = 0;
    binFormat.hasColRowtCoords = 0;
    binFormat.QDataType = "none";
    binFormat.QOutputLength = QVecSize;

    RunningParams running;

    std::cout << "Node " << nodeID << " Load " << (outputTree.getAllParamsFilePath(dirname, unit_name) + ".json") << std::endl;
    outputTree.readAllParamsJson(outputTree.getAllParamsFilePath(dirname, unit_name) + ".json", binFormat, running);
    flow = outputTree.getFlowParams<T>();
    init(outputTree.getComputeUnitParams());
    
    std::string filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "Q");
    FILE *fpQ = fopen_read(filePath);
    if (fpQ) {
        fread(Q, sizeof(QVec<T, QVecSize>), size, fpQ);
        fclose(fpQ);
    } else {
        std::cerr << "File " << filePath << " couldn't be read. " << std::strerror(errno) << std::endl;
    }

    
    filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "F");
    FILE *fpF = fopen_read(filePath);
    if (fpF) {
        fread(F, sizeof(Force<T>), size, fpF);
        fclose(fpF);
    } else {
        std::cerr << "File " << filePath << " couldn't be read. " << std::strerror(errno) << std::endl;
    }

    
    filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "Nu");
    FILE *fpNu = fopen_read(filePath);
    if (fpNu) {
        fread(Nu, sizeof(T), size, fpNu);
        fclose(fpNu);
    } else {
        std::cerr << "File " << filePath << " couldn't be read. " << std::strerror(errno) << std::endl;
    }

}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::checkpoint_write(std::string unit_name, RunningParams run){
    

    std::string dirname = outputTree.getCheckpointDirName(run);
    
    
    BinFileParams binFormat;
    binFormat.filePath = outputTree.getAllParamsFilePath(dirname, unit_name);
    binFormat.structName = "checkpoint";
    binFormat.binFileSizeInStructs = (Json::UInt64)size;
    binFormat.coordsType = "none";
    binFormat.hasGridtCoords = 0;
    binFormat.hasColRowtCoords = 0;
    binFormat.QDataType = "none";
    binFormat.QOutputLength = QVecSize;

    std::cout << "Node " << nodeID << " Save " << (binFormat.filePath + ".json") << std::endl;
    outputTree.setRunningParams(run);
    outputTree.writeAllParamsJson(binFormat, run);
    
    std::string filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "Q");
    FILE *fpQ = fopen_write(filePath);
    fwrite(Q, sizeof(QVec<T, QVecSize>), size, fpQ);
    fclose(fpQ);
    
    
    filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "F");
    FILE *fpF = fopen_write(filePath);
    fwrite(F, sizeof(Force<T>), size, fpF);
    fclose(fpF);
    
    
    filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "Nu");
    FILE *fpNu = fopen_write(filePath);
    fwrite(Nu, sizeof(T), size, fpNu);
    fclose(fpNu);

}













template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::index(tNi i, tNi j, tNi k)
{
#ifdef DEBUG
    if ((i>=xg) || (j>=yg) || (k>=zg)) {
        std::cout << "Index Error  " << i <<" "<< xg <<" "<< j <<" "<< yg <<" "<< k <<" "<< zg << std::endl;
        exit(1);
    }
#endif
    return i * (yg * zg) + (j * zg) + k;
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::indexPlusGhost(tNi i, tNi j, tNi k)
{
#ifdef DEBUG
    if (((i + ghost)>=xg) || ((j + ghost)>=yg) || ((k + ghost)>=zg)) {
        std::cout << "Index Error  " << i <<" "<< xg <<" "<< j <<" "<< yg <<" "<< k <<" "<< zg << std::endl;
        exit(1);
    }
#endif
    return (i + ghost) * (yg * zg) + ((j + ghost) * zg) + (k + ghost);
}











//NO DIRECTION
// 0  0  0
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ00(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + (j * zg) + k;
}


//RIGHT DIRECTION
// +1  0  0
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ01(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + (j * zg) + k;
}


//LEFT DIRECTION
// -1  0  0
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ02(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + (j * zg) + k;
}


//UP DIRECTION
//  0 +1  0
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ03(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j + 1) * zg) + k;
}


//DOWN DIRECTION
//  0 -1  0
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ04(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j - 1) * zg) + k;
}


//BACKWARD DIRECTION
//  0  0 +1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ05(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + (j * zg) + (k + 1);
}


//FORWARD DIRECTION
//  0  0 -1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ06(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + (j * zg) + (k - 1);
}


//RIGHT_UP DIRECTION
// +1 +1  0
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ07(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j + 1) * zg) + k;
}


//LEFT_DOWN DIRECTION
// -1 -1  0
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ08(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j - 1) * zg) + k;
}


//RIGHT_BACKWARD DIRECTION
// +1  0 +1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ09(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + (j * zg) + (k + 1);
}


//LEFT_FORWARD DIRECTION
// -1  0 -1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ10(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + (j * zg) + (k - 1);
}


//UP_BACKWARD DIRECTION
//  0 +1 +1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ11(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j + 1) * zg) + (k + 1);
}


//DOWN_FORWARD DIRECTION
//  0 -1 -1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ12(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j - 1) * zg) + (k - 1);
}


//RIGHT_DOWN DIRECTION
// +1 -1  0
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ13(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j - 1) * zg) + k;
}


//LEFT_UP DIRECTION
// -1 +1  0
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ14(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j + 1) * zg) + k;
}


//RIGHT_FORWARD DIRECTION
// +1  0 -1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ15(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + (j * zg) + (k - 1);
}


//LEFT_BACKWARD DIRECTION
// -1  0 +1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ16(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + (j * zg) + (k + 1);
}


//UP_FORWARD DIRECTION
//  0 +1 -1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ17(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j + 1) * zg) + (k - 1);
}


//DOWN_BACKWARD DIRECTION
//  0 -1 +1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ18(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j - 1) * zg) + (k + 1);
}


//RIGHT_UP_BACKWARD DIRECTION
// +1 +1 +1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ19(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j + 1) * zg) + (k + 1);
}


//LEFT_DOWN_FORWARD DIRECTION
// -1 -1 -1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ20(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j - 1) * zg) + (k - 1);
}


//RIGHT_UP_FORWARD DIRECTION
// +1 +1 -1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ21(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j + 1) * zg) + (k - 1);
}


//LEFT_DOWN_BACKWARD DIRECTION
// -1 -1 +1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ22(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j - 1) * zg) + (k + 1);
}


//RIGHT_DOWN_BACKWARD DIRECTION
// +1 -1 +1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ23(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j - 1) * zg) + (k + 1);
}


//LEFT_UP_FORWARD DIRECTION
// -1 +1 -1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ24(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j + 1) * zg) + (k - 1);
}


//LEFT_UP_BACKWARD DIRECTION
// -1 +1 +1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ25(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j + 1) * zg) + (k + 1);
}


//RIGHT_DOWN_FORWARD DIRECTION
// +1 -1 -1
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::dirnQ26(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j - 1) * zg) + (k - 1);
}
