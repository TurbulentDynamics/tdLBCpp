#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "ComputeUnit.h"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
__global__ void setFToZeroWhenOIsZero(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu){

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= cu.xg || j >= cu.yg || k >= cu.zg) {
        return;
    }

    //pos =  offsetb + offsets + k;
    if (cu.O[cu.index(i,j,k)] == 0) {
        cu.F[cu.index(i,j,k)].x = 0.0;
        cu.F[cu.index(i,j,k)].y = 0.0;
        cu.F[cu.index(i,j,k)].z = 0.0;
    } else {
        //Set it back to 0
        cu.O[cu.index(i,j,k)] = 0;
    }//endif
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
__global__ void forcing(ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, streamingType> &cu, PosPolar<tNi, T> *geom, size_t geomSize, T alfa, T beta){
    using AF = AccessField<T, QVecSize, MemoryLayout, collisionType, streamingType>;

    tNi geomIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (geomIndex >= geomSize) {
        return;
    }

    auto &g = geom[geomIndex];

    T ppp[3][3];

    tNi i = g.i + cu.ghost;
    tNi j = g.j + cu.ghost;
    tNi k = g.k + cu.ghost;


    smoothedDeltaFunction(g.iCartFraction, g.kCartFraction, ppp);


    T rhoSum = 0.0;
    T xSum = 0.0;
    T ySum = 0.0;
    T zSum = 0.0;



    for (tNi k1 = -1; k1<=1; k1++){
        for (tNi i1 = -1; i1<=1; i1++){

            tNi i2 = i + i1;
            tNi j2 = j;
            tNi k2 = k + k1;

            if (i2 == 0)   i2 = cu.xg1;
            if (i2 == cu.xg0) i2 = 1;
            if (k2 == 0)   k2 = cu.zg1;
            if (k2 == cu.zg0) k2 = 1;

            QVec<T, QVecSize> q = AF::read(cu, i2, j2, k2);
            T rho = q[M01];

            Force<T> localForce = cu.F[cu.index(i2,j2,k2)];

            T x = q[M02] + 0.5 * localForce.x;
            T y = q[M03] + 0.5 * localForce.y;
            T z = q[M04] + 0.5 * localForce.z;


            //adding the density of a nearby point using a weight (in ppp)
            rhoSum += ppp[i1+1][k1+1] * rho;

            //adding the velocity of a nearby point using a weight (in ppp)
            xSum += ppp[i1+1][k1+1] * x;
            ySum += ppp[i1+1][k1+1] * y;
            zSum += ppp[i1+1][k1+1] * z;
        }
    }//endfor  j1, k1


    //calculating the difference between the actual (weighted) speed and
    //the required (no-slip) velocity
    xSum -= rhoSum * g.uDelta;
    ySum -= rhoSum * g.vDelta;
    zSum -= rhoSum * g.wDelta;


    for (tNi k1 = -1; k1<=1; k1++){
        for (tNi i1 = -1; i1<=1; i1++){

            tNi i2 = i + i1;
            tNi j2 = j;
            tNi k2 = k + k1;


            if (i2 == 0)   i2 = cu.xg1;
            if (i2 == cu.xg0) i2 = 1;
            if (k2 == 0)   k2 = cu.zg1;
            if (k2 == cu.zg0) k2 = 1;



            Force<T> localForce = cu.F[cu.index(i2,j2,k2)];

            cu.F[cu.index(i2,j2,k2)].x = alfa * localForce.x - beta * ppp[i1+1][k1+1] * xSum;
            cu.F[cu.index(i2,j2,k2)].y = alfa * localForce.y - beta * ppp[i1+1][k1+1] * ySum;
            cu.F[cu.index(i2,j2,k2)].z = alfa * localForce.z - beta * ppp[i1+1][k1+1] * zSum;


            cu.O[cu.index(i2,j2,k2)] = 1;

        }
    }//endfor  j1, k1
}//end of func



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
__global__ void forcingDUMMY(ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, streamingType> &cu, PosPolar<tNi, T> *geom, size_t geomSize){
    using AF = AccessField<T, QVecSize, MemoryLayout, collisionType, streamingType>;

    tNi geomIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (geomIndex >= geomSize) {
        return;
    }

    auto &g = geom[geomIndex];

    tNi i = g.i + cu.ghost;
    tNi j = g.j + cu.ghost;
    tNi k = g.k + cu.ghost;


    for (tNi k1 = -1; k1<=1; k1++){
        for (tNi i1 = -1; i1<=1; i1++){

            tNi i2 = i + i1;
            tNi j2 = j;
            tNi k2 = k + k1;


            if (i2 == 0)   i2 = cu.xg1;
            if (i2 == cu.xg0) i2 = 1;
            if (k2 == 0)   k2 = cu.zg1;
            if (k2 == cu.zg0) k2 = 1;

            cu.O[cu.index(i2,j2,k2)] = 1;

    }}//endfor  j1, k1

}//end of func

