//
//  CollisionEgglesSomers.cuh
//  tdLBcpp
//
//  Created by Niall Ó Broin on 2021/09/03.
//

#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "ComputeUnit.h"



__global
template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
void ComputeUnitCollision<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>::collision(){
    using AF = AccessField<T, QVecSize, MemoryLayout, streamingType>;

    //kinematic viscosity.
    T b = 1.0 / (1.0 + 6 * flow.nu);
    T c = 1.0 - 6 * flow.nu;


    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i>xg1 || j>yg1 || k>zg1 || i<1 || j<1 || k<1) return;




    Force<T> f = F[index(i,j,k)];


    //TODO Change this to m, but write to q, notation only
    QVec<T, QVecSize> q = AF::read(*this, i, j, k);


    Velocity<T> u = q.velocity(f);

    QVec<T, QVecSize> alpha;


//TODO

    AF::write(*this, q, i, j, k);


}



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
void ComputeUnitCollision<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>::moments(){

    using QVecAcc = QVecAccess<T, QVecSize, MemoryLayout>;

    tNi i = blockIdx.x * blockDim.x + threadIdx.x;
    tNi j = blockIdx.y * blockDim.y + threadIdx.y;
    tNi k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i>xg1 || j>yg1 || k>zg1 || i<1 || j<1 || k<1) return;



    QVecAcc q = Q[index(i, j, k)];


    QVec<T, QVecSize> m = Q[index(i, j, k)];

//TODO



}//end of func




