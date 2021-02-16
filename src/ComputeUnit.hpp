//
//  ComputeGroup.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"

template <typename T, int QVecSize>
ComputeUnit<T, QVecSize>::ComputeUnit(
	tNi idi,
	tNi idj,
	tNi idk,
	tNi x,
	tNi y,
	tNi z,
	tNi ghost,
	FlowParams<T> flow
) :
	idi(idi),
	idj(idj),
	idk(idk),
	x(x),
	y(y),
	z(z),
	ghost(ghost),
	flow(flow)
{
    xg = x + 2 * ghost;
    yg = y + 2 * ghost;
    zg = z + 2 * ghost;

    size = size_t(xg) * yg * zg;

    Q = new QVec<T, QVecSize>[size];
    F = new Force<T>[size];
        𝜈 = new T[size];
    O = new bool[size];


#if WITH_GPU == 1
    checkCudaErrors(cudaSetDevice(0));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    unsigned long long GPUmem = prop.totalGlobalMem;

    if (((sizeof(QVec<T, QVecSize>) + sizeof(Force<T>) + sizeof(T)) * size) > GPUmem){
        std::cout << "Cannot allocate device on GPU." << std::endl;
        exit(1);
    }

    checkCudaErrors(cudaMalloc((void **)&devN, sizeof(QVec<T, QVecSize>) * size));
    checkCudaErrors(cudaMalloc((void **)&devF, sizeof(Force<T>) * size));
    checkCudaErrors(cudaMalloc((void **)&devNue, sizeof(T) * size));

    int threads_per_warp = 32;
    int max_threads_per_block = 512;

    int xthreads_per_block = 8;
    int ythreads_per_block = 8;
    int zthreads_per_block = 8;

    dim3 threadsPerBlock(xthreads_per_block, ythreads_per_block, zthreads_per_block);

    int block_in_x_dirn = xg / threadsPerBlock.x + (xg % xthreads_per_block != 0);
    int block_in_y_dirn = zg / threadsPerBlock.y + (yg % ythreads_per_block != 0);
    int block_in_z_dirn = zg / threadsPerBlock.z + (zg % zthreads_per_block != 0);

    dim3 numBlocks(block_in_x_dirn, block_in_y_dirn, block_in_z_dirn);

    std::cout << "threads_per_block" << threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z << std::endl;
    std::cout << "numBlocks" << numBlocks.x << ", " << numBlocks.y << ", " << numBlocks.z << std::endl;
#endif
};

template <typename T, int QVecSize>
ComputeUnit<T, QVecSize>::~ComputeUnit()
{
#if WITH_GPU == 1
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaFree(devN));
    checkCudaErrors(cudaFree(devF));
    checkCudaErrors(cudaFree(devNue));
#endif
}


template <typename T, int QVecSize>
tNi ComputeUnit<T, QVecSize>::index(tNi i, tNi j, tNi k)
{
#ifdef DEBUG
    if ((i>=xg) || (j>=yg) || (k>=zg)) {
        std::cout << "Index Error" << i <<" "<< xg <<" "<< j <<" "<< yg <<" "<< k <<" "<< zg << std::endl;
        exit(1);
    }
#endif
    return i * (yg * zg) + (j * zg) + k;
}

template <typename T, int QVecSize>
Velocity<T> ComputeUnit<T, QVecSize>::getVelocity(tNi i, tNi j, tNi k){
    return Q[index(i, j, k)].velocity();
};

template <typename T, int QVecSize>
Velocity<T> ComputeUnit<T, QVecSize>::getVelocitySparseF(tNi i, tNi j, tNi k, Force<T> f){
    return Q[index(i, j, k)].velocity(f);
};


template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::streaming(Streaming scheme) {

    switch( scheme ) {
    case Streaming(Simple):
        streaming_simple(); break;
    case Streaming(Esotwist):
        streaming_esotwist(); break;
    }
}




template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::collision_EgglesSomers_LES(){

//    alf2[ 5] = (2.0 * (q[2] + 0.5 * f.x) * u.x - q[ 5]*c)*b * Nue.getNue(i,j,k);

}


template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::collision_EgglesSomers(){
  

    //kinematic viscosity.
    T b = 1.0 / (1.0 + 6 * flow.nu);
    T c = 1.0 - 6 * flow.nu;
        
    
    for (tNi i=ghost; i<xg - ghost; i++){
        for (tNi j=ghost; j<yg - ghost; j++){
            for (tNi k=ghost; k<zg - ghost; k++){


                Force<T> f = F[index(i,j,k)];

				QVec<T, QVecSize> q = Q[index(i, j, k)];

				Velocity<T> u = q.velocity(f);

                QVec<T, QVecSize> alf2;


                //0th order term
                alf2[ Q1] = q[Q1];


                //1st order term
                alf2[ Q2] = q[Q2] + f.x;
                alf2[ Q3] = q[Q3] + f.y;
                alf2[ Q4] = q[Q4] + f.z;

                //2nd order terms
                alf2[ Q5] = (2.0 * (q[Q2] + 0.5 * f.x) * u.x - q[ Q5]*c)*b;
                alf2[ Q6] = (2.0 * (q[Q2] + 0.5 * f.x) * u.y - q[ Q6]*c)*b;
                alf2[ Q7] = (2.0 * (q[Q3] + 0.5 * f.y) * u.y - q[ Q7]*c)*b;
                alf2[ Q8] = (2.0 * (q[Q2] + 0.5 * f.x) * u.z - q[ Q8]*c)*b;
                alf2[ Q9] = (2.0 * (q[Q3] + 0.5 * f.y) * u.z - q[ Q9]*c)*b;
                alf2[Q10] = (2.0 * (q[Q4] + 0.5 * f.z) * u.z - q[Q10]*c)*b;

                //3rd order terms
                alf2[Q11] =  -flow.g3 * q[Q11];
                alf2[Q12] =  -flow.g3 * q[Q12];
                alf2[Q13] =  -flow.g3 * q[Q13];
                alf2[Q14] =  -flow.g3 * q[Q14];
                alf2[Q15] =  -flow.g3 * q[Q15];
                alf2[Q16] =  -flow.g3 * q[Q16];

                //4th order terms
                alf2[Q17] = 0.0;
                alf2[Q18] = 0.0;


                // Start of invMoments, which is responsible for determining
                // the N-field (x) from alpha+ (alf2). It does this by using eq.
                // 12 in the article by Eggels and Somers (1995), which means
                // it's using the "filter matrix E" (not really present in the
                // code as matrix, but it's where the coefficients come from).

                for (int l=0;  l<QVecSize; l++) {
                    alf2[l] /= 24.0;
                }


                q[Q1]  = 2 * (alf2[Q1] + 2*alf2[Q2] + 1.5*alf2[Q5] - 1.5*alf2[Q7] - 1.5*alf2[Q10] - alf2[Q11] - alf2[Q13] + alf2[Q17] + alf2[Q18]);

                q[Q2]  = 2 * (alf2[Q1] + 2*alf2[Q3] - 1.5*alf2[Q5] + 1.5*alf2[Q7] - 1.5*alf2[Q10] - alf2[Q12] - alf2[Q14] + alf2[Q17] - alf2[Q18]);

                q[Q3]  = 2 * (alf2[Q1] - 2*alf2[Q2] + 1.5*alf2[Q5] - 1.5*alf2[Q7] - 1.5*alf2[Q10] + alf2[Q11] + alf2[Q13] + alf2[Q17] + alf2[Q18]);

                q[Q4]  = 2 * (alf2[Q1] - 2*alf2[Q3] - 1.5*alf2[Q5] + 1.5*alf2[Q7] - 1.5*alf2[Q10] + alf2[Q12] + alf2[Q14] + alf2[Q17] - alf2[Q18]);

                q[Q5]  = 2 * (alf2[Q1] + 2*alf2[Q4] - 1.5*alf2[Q5] - 1.5*alf2[Q7] + 1.5*alf2[Q10] - 2*alf2[Q15] - 2*alf2[Q17]);

                q[Q6]  = 2 * (alf2[Q1] - 2*alf2[Q4] - 1.5*alf2[Q5] - 1.5*alf2[Q7] + 1.5*alf2[Q10] + 2*alf2[Q15] - 2*alf2[Q17]);

                q[Q7]  = alf2[Q1] + 2*alf2[Q2] + 2*alf2[Q3] + 1.5*alf2[Q5] + 6*alf2[Q6] + 1.5*alf2[Q7] - 1.5*alf2[Q10] + 2*alf2[Q11] + 2*alf2[Q12] - 2*alf2[Q17];

                q[Q8]  = alf2[Q1] - 2*alf2[Q2] + 2*alf2[Q3] + 1.5*alf2[Q5] - 6*alf2[Q6] + 1.5*alf2[Q7] - 1.5*alf2[Q10] - 2*alf2[Q11] + 2*alf2[Q12] - 2*alf2[Q17];

                q[Q9]  = alf2[Q1] - 2*alf2[Q2] - 2*alf2[Q3] + 1.5*alf2[Q5] + 6*alf2[Q6] + 1.5*alf2[Q7] - 1.5*alf2[Q10] - 2*alf2[Q11] - 2*alf2[Q12] - 2*alf2[Q17];

                q[Q10] = alf2[Q1] + 2*alf2[Q2] - 2*alf2[Q3] + 1.5*alf2[Q5] - 6*alf2[Q6] + 1.5*alf2[Q7] - 1.5*alf2[Q10] + 2*alf2[Q11] - 2*alf2[Q12] - 2*alf2[Q17];

                q[Q11] = alf2[Q1] + 2*alf2[Q2] + 2*alf2[Q4] + 1.5*alf2[Q5] - 1.5*alf2[Q7] + 6*alf2[Q8] + 1.5*alf2[Q10] - alf2[Q11] + alf2[Q13] + alf2[Q15] - alf2[Q16] + alf2[Q17] - alf2[Q18];

                q[Q12] = alf2[Q1] - 2*alf2[Q2] + 2*alf2[Q4] + 1.5*alf2[Q5] - 1.5*alf2[Q7] - 6*alf2[Q8] + 1.5*alf2[Q10] + alf2[Q11] - alf2[Q13] + alf2[Q15] - alf2[Q16] + alf2[Q17] - alf2[Q18];

                q[Q13] = alf2[Q1] - 2*alf2[Q2] - 2*alf2[Q4] + 1.5*alf2[Q5] - 1.5*alf2[Q7] + 6*alf2[Q8] + 1.5*alf2[Q10] + alf2[Q11] - alf2[Q13] - alf2[Q15] + alf2[Q16] + alf2[Q17] - alf2[Q18];

                q[Q14] = alf2[Q1] + 2*alf2[Q2] - 2*alf2[Q4] + 1.5*alf2[Q5] - 1.5*alf2[Q7] - 6*alf2[Q8] + 1.5*alf2[Q10] - alf2[Q11] + alf2[Q13] - alf2[Q15] + alf2[Q16] + alf2[Q17] - alf2[Q18];

                q[Q15] = alf2[Q1] + 2*alf2[Q3] + 2*alf2[Q4] - 1.5*alf2[Q5] + 1.5*alf2[Q7] + 6*alf2[Q9] + 1.5*alf2[Q10] - alf2[Q12] + alf2[Q14] + alf2[Q15] + alf2[Q16] + alf2[Q17] + alf2[Q18];

                q[Q16] = alf2[Q1] - 2*alf2[Q3] + 2*alf2[Q4] - 1.5*alf2[Q5] + 1.5*alf2[Q7] - 6*alf2[Q9] + 1.5*alf2[Q10] + alf2[Q12] - alf2[Q14] + alf2[Q15] + alf2[Q16] + alf2[Q17] + alf2[Q18];

                q[Q17] = alf2[Q1] - 2*alf2[Q3] - 2*alf2[Q4] - 1.5*alf2[Q5] + 1.5*alf2[Q7] + 6*alf2[Q9] + 1.5*alf2[Q10] + alf2[Q12] - alf2[Q14] - alf2[Q15] - alf2[Q16] + alf2[Q17] + alf2[Q18];

                q[Q18] = alf2[Q1] + 2*alf2[Q3] - 2*alf2[Q4] - 1.5*alf2[Q5] + 1.5*alf2[Q7] - 6*alf2[Q9] + 1.5*alf2[Q10] - alf2[Q12] + alf2[Q14] - alf2[Q15] - alf2[Q16] + alf2[Q17] + alf2[Q18];


                Q[index(i, j, k)] = alf2;

            }
        }
    }
}


template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::setToZero(){
#if WITH_GPU == 1
    setToZero<<<numBlocks, threadsPerBlock>>>(devN, devF, xg, yg, zg, QVecSize);
#else
    for (tNi i=0; i<xg; i++){
        for (tNi j=0; j<yg; j++){
            for (tNi k=0; k<zg; k++){
            
                Q[index(i, j, k)].setToZero();
            }
        }
    }
#endif
};


template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::fillForTest(){
    
    if (xg>99 || yg>99 || zg>99) {
        std::cout << "Size too large for testing" << std::endl;
        exit(1);
    }
#if WITH_GPU == 1
    setToZero<<<numBlocks, threadsPerBlock>>>(devN, devF, xg, yg, zg, QVecSize);
#else
    for (tNi i=0; i<xg; i++){
        for (tNi j=0; j<yg; j++){
            for (tNi k=0; k<zg; k++){
                
                QVec<int, QVecSize> qTmp;

                for (int l=0; l<QVecSize; l++){
                    qTmp.q[l] = i * 1000000 + j * 10000 + k * 100 + l;
                }
                Q[index(i, j, k)].q = qTmp;
                
                F[index(i, j, k)].fx = 0;
                F[index(i, j, k)].fy = 1;
                F[index(i, j, k)].fz = 2;

                    𝜈[index(i, j, k)] = 1;

            }
        }
    }
#endif
};

template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::streaming_esotwist() {
	//TODO
}

template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::collision_Entropic() {
	//TODO
}

template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::collision(Collision scheme){
    switch( scheme ) {

        case Collision(EgglesSomers):
            collision_EgglesSomers(); break;

        case Collision(EgglesSomersLES):
            collision_EgglesSomers_LES(); break;

        case Collision(Entropic):
            collision_Entropic(); break;
    }
};

template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::streaming_simple(){
    for (tNi i=ghost; i<xg - ghost; i++){
        for (tNi j=ghost; j<yg - ghost; j++){
            for (tNi k=ghost; k<zg - ghost; k++){
                //DST    =  SRC
                Q[index(i, j, k)].q[O14] = Q[index(i+1, j,   k  )].q[O14];
                Q[index(i, j, k)].q[O16] = Q[index(i,   j+1, k  )].q[O16];
                Q[index(i, j, k)].q[O22] = Q[index(i,   j,   k+1)].q[O22];

                Q[index(i, j, k)].q[O11] = Q[index(i+1, j-1, k  )].q[O11];
                Q[index(i, j, k)].q[O17] = Q[index(i+1, j+1, k  )].q[O17];
                
                Q[index(i, j, k)].q[ O5] = Q[index(i+1, j,   k-1)].q[ O5];
                Q[index(i, j, k)].q[O23] = Q[index(i+1, j,   k+1)].q[O23];
                
                Q[index(i, j, k)].q[ O7] = Q[index(i,   j+1, k-1)].q[ O7];
                Q[index(i, j, k)].q[O25] = Q[index(i,   j+1, k+1)].q[O25];
            }
        }
    }


    for (tNi i=xg-ghost;  i>=ghost; i--) {
        for (tNi j=yg-ghost;  j>=ghost; j--) {
            for (tNi k=zg-ghost;  k>=ghost; k--) {
                //DST   =   SRC

                Q[index(i, j, k)].q[O12] = Q[index(i-1, j,   k  )].q[O12];
                Q[index(i, j, k)].q[O10] = Q[index(i,   j-1, k  )].q[O10];
                Q[index(i, j, k)].q[ O4] = Q[index(i,   j,   k-1)].q[ O4];

                Q[index(i, j, k)].q[ O9] = Q[index(i-1, j-1, k  )].q[ O9];
                Q[index(i, j, k)].q[O15] = Q[index(i-1, j+1, k  )].q[O15];
                
                Q[index(i, j, k)].q[ O3] = Q[index(i-1, j,   k-1)].q[ O3];
                Q[index(i, j, k)].q[O21] = Q[index(i-1, j,   k+1)].q[O21];
                
                Q[index(i, j, k)].q[ O1] = Q[index(i,   j-1, k-1)].q[ O1];
                Q[index(i, j, k)].q[O19] = Q[index(i,   j-1, k+1)].q[O19];
            }
        }
    }
}

template <typename T, int QVecSize>
std::string ComputeUnit<T, QVecSize>::get_checkpoint_filename(std::string dirname, std::string unit_name, std::string matrix){

    
    std::string path = dirname + "/checkpoint_grid." + std::to_string(idi) + "." + std::to_string(idj) + "." + std::to_string(idk) + ".";

    path += unit_name + "." + matrix;

    return path;
}

template <typename T, int QVecSize>
FILE* ComputeUnit<T, QVecSize>::fopen_read(std::string dirname, std::string unit_name, std::string matrix){

    std::string pathname = get_checkpoint_filename(dirname, unit_name, matrix);

    std::cout << "Node " << rank << " Load " << pathname << std::endl;

    return fopen(pathname.c_str(), "r");
}

template <typename T, int QVecSize>
FILE* ComputeUnit<T, QVecSize>::fopen_write(std::string dirname, std::string unit_name, std::string matrix){

    std::string pathname = get_checkpoint_filename(dirname, unit_name, matrix);

    std::cout << "Node " << rank << " Save " << pathname << std::endl;

    return fopen(pathname.c_str(), "w");
}





template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::checkpoint_read(std::string dirname, std::string unit_name){
    
    FILE *fpN = fopen_read(dirname, unit_name, "N");
    fread(Q, sizeof(QVec<T, QVecSize>), size, fpN);
    fclose(fpN);
    
    FILE *fpF = fopen_read(dirname, unit_name, "F");
    fread(F, sizeof(Force<T>), size, fpF);
    fclose(fpF);
    
    FILE *fpNue = fopen_read(dirname, unit_name, "Nue");
    fread(    𝜈, sizeof(T), size, fpNue);
    fclose(fpNue);
}

template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::checkpoint_write(std::string dirname, std::string unit_name){
    
    FILE *fpN = fopen_read(dirname, unit_name, "F");
    fread(Q, sizeof(QVec<T, QVecSize>), size, fpN);
    fclose(fpN);
    
    FILE *fpF = fopen_read(dirname, unit_name, "F");
    fread(F, sizeof(Force<T>), size, fpF);
    fclose(fpF);
    
    FILE *fpNue = fopen_read(dirname, unit_name, "F");
    fread(    𝜈, sizeof(T), size, fpNue);
    fclose(fpNue);
}

