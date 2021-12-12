#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "ComputeUnit.h"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
__device__ T calcVorticity(ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType> &cu, tNi i, tNi j, tNi k) {
    using AF = AccessField<T, QVecSize, MemoryLayout, collisionType, streamingType>;

    if (cu.devExcludeOutputPoints[cu.index(i,j,k)] == true) {
        return (T)0;
    }


    QVec<T, QVecSize> qDirnQ05 = AF::read(cu, i, j, k + 1);
    QVec<T, QVecSize> qDirnQ06 = AF::read(cu, i, j, k - 1);
    T uxy = T(0.5) * (qDirnQ05.velocity().x - qDirnQ06.velocity().x);
    QVec<T, QVecSize> qDirnQ03 = AF::read(cu, i, j + 1, k);
    QVec<T, QVecSize> qDirnQ04 = AF::read(cu, i, j - 1, k);
    T uxz = T(0.5) * (qDirnQ03.velocity().x - qDirnQ04.velocity().x);

    QVec<T, QVecSize> qDirnQ01 = AF::read(cu, i + 1, j, k);
    QVec<T, QVecSize> qDirnQ02 = AF::read(cu, i - 1, j, k);
    T uyx = T(0.5) * (qDirnQ01.velocity().y - qDirnQ02.velocity().y);
    T uyz = T(0.5) * (qDirnQ03.velocity().y - qDirnQ04.velocity().y);


    T uzx = T(0.5) * (qDirnQ01.velocity().z - qDirnQ02.velocity().z);
    T uzy = T(0.5) * (qDirnQ05.velocity().z - qDirnQ06.velocity().z);


    T uxyuyx = uxy - uyx;
    T uyzuzy = uyz - uzy;
    T uzxuxz = uzx - uxz;

    return T(log(T(uyzuzy * uyzuzy + uzxuxz * uzxuxz + uxyuyx * uxyuyx)));
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
__global__ void calcVorticityXZ(ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType> &cu, tNi j) {
    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > cu.xg1 || j > cu.yg1 || k > cu.zg1 || i < 1 || j < 1 || k < 1) {
        return;
    }

    cu.VortXZ[cu.xg * k + i] = calcVorticity(cu, i, j, k);

}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
__global__ void calcVorticityXY(ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType> &cu, tNi k) {
    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > cu.xg1 || j > cu.yg1 || k > cu.zg1 || i < 1 || j < 1 || k < 1) {
        return;
    }

    cu.VortXY[cu.xg * j + i] = calcVorticity(cu, i, j, k);

}