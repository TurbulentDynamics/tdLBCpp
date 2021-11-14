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

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
__global__ void setQToZero(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu){

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= cu.xg || j >= cu.yg || k >= cu.zg) {
        return;
    }

#pragma unroll
    for (tNi l = 0; l < QVecSize; l++){
        cu.Q[cu.index(i,j,k)].q[l] = 0.0;
    }
};



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
__global__ void setRhoTo(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, T initialRho){

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= cu.xg || j >= cu.yg || k >= cu.zg) {
        return;
    }

    cu.Q[cu.index(i, j, k)].q[MRHO] = initialRho;
};


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
__global__ void setForceToZero(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, T initialRho){

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= cu.xg || j >= cu.yg || k >= cu.zg) {
        return;
    }

    cu.F[cu.index(i, j, k)].x = 0.0;
    cu.F[cu.index(i, j, k)].y = 0.0;
    cu.F[cu.index(i, j, k)].z = 0.0;

};




template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
__global__ void setNuToZero(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, T initialRho){

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= cu.xg || j >= cu.yg || k >= cu.zg) {
        return;
    }

    cu.Nu[cu.index(i, j, k)] = 0.0;
};



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
__global__ void setOToZero(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu){

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= cu.xg || j >= cu.yg || k >= cu.zg) {
        return;
    }

    cu.O[cu.index(i, j, k)] = false;

};
