#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "ComputeUnit.h"

template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
__global__ void bounceBackEdges(ComputeUnitBase<T, NVecSize, MemoryLayout> &cu, int direction){

    tNi i, j, k;
    tNi index = blockIdx.x * blockDim.x + threadIdx.x;

    if (direction == Q05) {
        if ((index < 1) || (index > cu.zg1)) {
            return;
        }
        k = index;

        i = 0;
        j = 0;
        cu.Q[cu.index(i,j,index)].q[Q07] = cu.Q[cu.dirnQ07(i, j, k)].q[Q08];

        i = cu.xg0;
        j = 0;
        cu.Q[cu.index(i,j,k)].q[Q14] = cu.Q[cu.dirnQ14(i, j, k)].q[Q13];

        i = 0;
        j = cu.yg0;
        cu.Q[cu.index(i,j,k)].q[Q13] = cu.Q[cu.dirnQ13(i, j, k)].q[Q14];

        i = cu.xg0;
        j = cu.yg0;
        cu.Q[cu.index(i,j,k)].q[Q08] = cu.Q[cu.dirnQ08(i, j, k)].q[Q07];
    }


    if (direction == Q03) {
        if ((index < 1) || (index > cu.yg1)) {
            return;
        }
        j = index;

        i = 0;
        k = 0;
        cu.Q[cu.index(i,j,k)].q[Q09] = cu.Q[cu.dirnQ09(i, j, k)].q[Q10];

        i = 0;
        k = cu.zg0;
        cu.Q[cu.index(i,j,k)].q[Q15] = cu.Q[cu.dirnQ15(i, j, k)].q[Q16];

        i = cu.xg0;
        k = cu.zg0;
        cu.Q[cu.index(i,j,k)].q[Q10] = cu.Q[cu.dirnQ10(i, j, k)].q[Q09];

        i = cu.xg0;
        k = 0;
        cu.Q[cu.index(i,j,k)].q[Q16] = cu.Q[cu.dirnQ16(i, j, k)].q[Q15];
    }

    if (direction == Q01) {
        if ((index < 1) || (index > cu.xg1)) {
            return;
        }
        i = index;
        j = 0;
        k = 0;
        cu.Q[cu.index(i,j,k)].q[Q11] = cu.Q[cu.dirnQ11(i, j, k)].q[Q12];

        j = 0;
        k = cu.zg0;
        cu.Q[cu.index(i,j,k)].q[Q17] = cu.Q[cu.dirnQ17(i, j, k)].q[Q18];

        j = cu.yg0;
        k = cu.zg0;
        cu.Q[cu.index(i,j,k)].q[Q12] = cu.Q[cu.dirnQ12(i, j, k)].q[Q11];

        j = cu.yg0;
        k = 0;
        cu.Q[cu.index(i,j,k)].q[Q18] = cu.Q[cu.dirnQ18(i, j, k)].q[Q17];
    }
}



template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
__global__ void bounceBackBoundaryRight(ComputeUnitBase<T, NVecSize, MemoryLayout> &cu){
    tNi j = blockIdx.x * blockDim.x + threadIdx.x;
    tNi k = blockIdx.y * blockDim.y + threadIdx.y;
    if ((j < 1) || (j > cu.yg1)) {
        return;
    }
    if ((k < 1) || (k > cu.zg1)) {
        return;
    }

    tNi i = 0;

    cu.Q[cu.index(i,j,k)].q[Q01] = cu.Q[cu.dirnQ01(i, j, k)].q[Q02];
    cu.Q[cu.index(i,j,k)].q[Q07] = cu.Q[cu.dirnQ07(i, j, k)].q[Q08];
    cu.Q[cu.index(i,j,k)].q[Q13] = cu.Q[cu.dirnQ13(i, j, k)].q[Q14];
    cu.Q[cu.index(i,j,k)].q[Q09] = cu.Q[cu.dirnQ09(i, j, k)].q[Q10];
    cu.Q[cu.index(i,j,k)].q[Q15] = cu.Q[cu.dirnQ15(i, j, k)].q[Q16];
}


template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
__global__ void bounceBackBoundaryLeft(ComputeUnitBase<T, NVecSize, MemoryLayout> &cu){
    tNi j = blockIdx.x * blockDim.x + threadIdx.x;
    tNi k = blockIdx.y * blockDim.y + threadIdx.y;
    if ((j < 1) || (j > cu.yg1)) {
        return;
    }
    if ((k < 1) || (k > cu.zg1)) {
        return;
    }

    tNi i = cu.xg0;

    cu.Q[cu.index(i,j,k)].q[Q02] = cu.Q[cu.dirnQ02(i, j, k)].q[Q01];
    cu.Q[cu.index(i,j,k)].q[Q08] = cu.Q[cu.dirnQ08(i, j, k)].q[Q07];
    cu.Q[cu.index(i,j,k)].q[Q14] = cu.Q[cu.dirnQ14(i, j, k)].q[Q13];
    cu.Q[cu.index(i,j,k)].q[Q10] = cu.Q[cu.dirnQ10(i, j, k)].q[Q09];
    cu.Q[cu.index(i,j,k)].q[Q16] = cu.Q[cu.dirnQ16(i, j, k)].q[Q15];
}


template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
__global__ void bounceBackBoundaryUp(ComputeUnitBase<T, NVecSize, MemoryLayout> &cu){
    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi k = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < 1) || (i > cu.xg1)) {
        return;
    }
    if ((k < 1) || (k > cu.zg1)) {
        return;
    }

    tNi j = 0;

    cu.Q[cu.index(i,j,k)].q[Q03] = cu.Q[cu.dirnQ03(i, j, k)].q[Q04];
    cu.Q[cu.index(i,j,k)].q[Q07] = cu.Q[cu.dirnQ07(i, j, k)].q[Q08];
    cu.Q[cu.index(i,j,k)].q[Q14] = cu.Q[cu.dirnQ14(i, j, k)].q[Q13];
    cu.Q[cu.index(i,j,k)].q[Q11] = cu.Q[cu.dirnQ11(i, j, k)].q[Q12];
    cu.Q[cu.index(i,j,k)].q[Q17] = cu.Q[cu.dirnQ17(i, j, k)].q[Q18];
}


template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
__global__ void bounceBackBoundaryDown(ComputeUnitBase<T, NVecSize, MemoryLayout> &cu){
    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi k = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < 1) || (i > cu.xg1)) {
        return;
    }
    if ((k < 1) || (k > cu.zg1)) {
        return;
    }

    tNi j = cu.yg0;

    cu.Q[cu.index(i,j,k)].q[Q04] = cu.Q[cu.dirnQ04(i, j, k)].q[Q03];
    cu.Q[cu.index(i,j,k)].q[Q08] = cu.Q[cu.dirnQ08(i, j, k)].q[Q07];
    cu.Q[cu.index(i,j,k)].q[Q13] = cu.Q[cu.dirnQ13(i, j, k)].q[Q14];
    cu.Q[cu.index(i,j,k)].q[Q12] = cu.Q[cu.dirnQ12(i, j, k)].q[Q11];
    cu.Q[cu.index(i,j,k)].q[Q18] = cu.Q[cu.dirnQ18(i, j, k)].q[Q17];
}


template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
__global__ void bounceBackBoundaryBackward(ComputeUnitBase<T, NVecSize, MemoryLayout> &cu){
    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < 1) || (i > cu.xg1)) {
        return;
    }
    if ((j < 1) || (j > cu.yg1)) {
        return;
    }

    tNi k = 0;


    cu.Q[cu.index(i,j,k)].q[Q05] = cu.Q[cu.dirnQ05(i, j, k)].q[Q06];
    cu.Q[cu.index(i,j,k)].q[Q09] = cu.Q[cu.dirnQ09(i, j, k)].q[Q10];
    cu.Q[cu.index(i,j,k)].q[Q16] = cu.Q[cu.dirnQ16(i, j, k)].q[Q15];
    cu.Q[cu.index(i,j,k)].q[Q11] = cu.Q[cu.dirnQ11(i, j, k)].q[Q12];
    cu.Q[cu.index(i,j,k)].q[Q18] = cu.Q[cu.dirnQ18(i, j, k)].q[Q17];

}


template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
__global__ void bounceBackBoundaryForward(ComputeUnitBase<T, NVecSize, MemoryLayout> &cu){
    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < 1) || (i > cu.xg1)) {
        return;
    }
    if ((j < 1) || (j > cu.yg1)) {
        return;
    }

    tNi k = cu.zg0;

    cu.Q[cu.index(i,j,k)].q[Q06] = cu.Q[cu.dirnQ06(i, j, k)].q[Q05];
    cu.Q[cu.index(i,j,k)].q[Q10] = cu.Q[cu.dirnQ10(i, j, k)].q[Q09];
    cu.Q[cu.index(i,j,k)].q[Q15] = cu.Q[cu.dirnQ15(i, j, k)].q[Q16];
    cu.Q[cu.index(i,j,k)].q[Q12] = cu.Q[cu.dirnQ12(i, j, k)].q[Q11];
    cu.Q[cu.index(i,j,k)].q[Q17] = cu.Q[cu.dirnQ17(i, j, k)].q[Q18];
}
