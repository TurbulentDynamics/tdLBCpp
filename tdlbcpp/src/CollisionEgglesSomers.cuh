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




    if (flow.useLES == 1){
        T fct = 3.0 / (q.q[M01] * (1.0 + 6.0 * (Nu[index(i,j,k)] + flow.nu)));

        //calculating the derivatives for x, y and z
        T dudx = fct * ((q.q[M02] + 0.5 * F[index(i,j,k)].x * u.x - q.q[M05]));
        T dvdy = fct * ((q.q[M03] + 0.5 * F[index(i,j,k)].y * u.y - q.q[M07]));
        T dwdz = fct * ((q.q[M04] + 0.5 * F[index(i,j,k)].z * u.z - q.q[M10]));

        T divv = dudx + dvdy + dwdz;


        //calculating the cross terms, used for the shear matrix
        T dudypdvdx = 2 * fct * ((q.q[M03]) + 0.5 * F[index(i,j,k)].y * u.x - q.q[M06]);
        T dudzpdwdx = 2 * fct * ((q.q[M04]) + 0.5 * F[index(i,j,k)].z * u.x - q.q[M08]);
        T dvdzpdwdy = 2 * fct * ((q.q[M04]) + 0.5 * F[index(i,j,k)].z * u.y - q.q[M09]);


        //calculating sh (the resolved deformation rate, S^2)
        T sh = 2 * pow(dudx,2) + 2 * pow(dvdy,2) + 2 * pow(dwdz,2) + pow(dudypdvdx,2) + pow(dudzpdwdx,2) + pow(dvdzpdwdy,2) - (2.0/3.0) * pow(divv,2);


        //calculating eddy viscosity:
        //nu_t = (lambda_mix)^2 * sqrt(S^2)     (Smagorinsky)
        Nu[index(i,j,k)] = flow.cs0 * flow.cs0 * sqrt(fabs(sh));



        // Viscosity is adjusted only for LES, because LES uses a
        // subgrid-adjustment model for turbulence that's too small to
        // be captured in the regular cells. This adjustment is
        // performed by adding the eddy viscosity to the viscosity.
        // This model is called the Smagorinsky model, however this
        // implementation is slightly different, as explained by
        // Somers (1993) -> low strain rates do not excite the
        // eddy viscosity.

        T nut = Nu[index(i,j,k)] + flow.nu;
        b = 1.0 / (1.0 + 6 * nut);
        c = 1.0 - 6 * nut;

    }//end of LES

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




