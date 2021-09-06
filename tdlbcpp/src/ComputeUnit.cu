//
//  ComputeUnit.cpp
//  tdLBCpp
//
//  Created by Niall Ó Broin on 06/09/2021.
//

#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "ComputeUnit.h"


__global__
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::setQToZero(){

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;

#pragma unroll
    for (tNi l = 0; l < QVecSize; l++){
        Q[index(i,j,k)].q[l] = 0.0;
    }
};



__global__
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::setRhoTo(T initialRho){

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;

    Q[index(i, j, k)].q[MRHO](initialRho);
};


__global__
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::setForceToZero(T initialRho){

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;

    F[index(i, j, k)].x = 0.0;
    F[index(i, j, k)].y = 0.0;
    F[index(i, j, k)].z = 0.0;

};




__global__
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::setNuToZero(T initialRho){

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;

    Nu[index(i, j, k)] = 0.0;
};



__global__
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::setOToZero(T initialRho){

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;

    O[index(i, j, k)] = false;

};
