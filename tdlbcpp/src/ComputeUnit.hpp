//
//  ComputeUnit.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"
#include "DiskOutputTree.h"


//
//template <typename T, int QVecSize>
//ComputeUnit<T, QVecSize>::ComputeUnit(ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree):flow(flow), outputTree(outputTree){

    
template <typename T, int QVecSize>
ComputeUnit<T, QVecSize>::ComputeUnit(ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree):flow(flow), outputTree(outputTree){

    evenStep = true;
    
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


    size = size_t(xg) * yg * zg;
    
    
    Q = new QVec<T, QVecSize>[size];
    F = new Force<T>[size];
    Nu = new T[size];
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
    checkCudaErrors(cudaMalloc((void **)&devNu, sizeof(T) * size));

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
    checkCudaErrors(cudaFree(devNu));
#endif
}




template <typename T, int QVecSize>
template <Streaming streaming>
void ComputeUnit<T, QVecSize>::collision(Collision scheme){
    switch( scheme ) {

        case Collision(EgglesSomers):
            collision_EgglesSomers<streaming>(); break;

        case Collision(EgglesSomersLES):
            collision_EgglesSomers_LES(); break;

        case Collision(Entropic):
            collision_Entropic(); break;
    }
};


template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::streaming(Streaming scheme) {

    switch( scheme ) {
    case Streaming(Simple):
        streamingNieve(); break;
    case Streaming(Esotwist):
        streaming_esotwist(); break;
    }
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
FILE* ComputeUnit<T, QVecSize>::fopen_read(std::string filePath){

    std::cout << "Node " << mpiRank << " Load " << filePath << std::endl;

    return fopen(filePath.c_str(), "r");
}

template <typename T, int QVecSize>
FILE* ComputeUnit<T, QVecSize>::fopen_write(std::string filePath){

    std::cout << "Node " << mpiRank << " Save " << filePath << std::endl;

    return fopen(filePath.c_str(), "w");
}





template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::checkpoint_read(std::string dirname, std::string unit_name){
    
    
    std::string filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "Q");
    FILE *fpQ = fopen_read(filePath);
    fread(Q, sizeof(QVec<T, QVecSize>), size, fpQ);
    fclose(fpQ);

    
    filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "F");
    FILE *fpF = fopen_read(filePath);
    fread(F, sizeof(Force<T>), size, fpF);
    fclose(fpF);

    
    filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "Nu");
    FILE *fpNu = fopen_read(filePath);
    fread(Nu, sizeof(T), size, fpNu);
    fclose(fpNu);

}


template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::checkpoint_write(std::string unit_name, RunningParams run){
    

    std::string dirname = outputTree.getCheckpointDirName(run);
    
    
    BinFileParams binFormat;
    binFormat.filePath = dirname + "/AllParams";
    binFormat.structName = "checkpoint";
    binFormat.binFileSizeInStructs = (Json::UInt64)size;
    binFormat.coordsType = "none";
    binFormat.hasGridtCoords = 0;
    binFormat.hasColRowtCoords = 0;
    binFormat.QDataType = "none";
    binFormat.QOutputLength = QVecSize;

    
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













template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::index(tNi i, tNi j, tNi k)
{
#ifdef DEBUG
    if ((i>=xg) || (j>=yg) || (k>=zg)) {
        std::cout << "Index Error  " << i <<" "<< xg <<" "<< j <<" "<< yg <<" "<< k <<" "<< zg << std::endl;
        exit(1);
    }
#endif
    return i * (yg * zg) + (j * zg) + k;
}



//NO DIRECTION
// 0  0  0
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ000(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + (j * zg) + k;
}


//RIGHT DIRECTION
// +1  0  0
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ1(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + (j * zg) + k;
}


//LEFT DIRECTION
// -1  0  0
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ2(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + (j * zg) + k;
}


//UP DIRECTION
//  0 +1  0
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ3(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j + 1) * zg) + k;
}


//DOWN DIRECTION
//  0 -1  0
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ4(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j - 1) * zg) + k;
}


//BACKWARD DIRECTION
//  0  0 +1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ5(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + (j * zg) + (k + 1);
}


//FORWARD DIRECTION
//  0  0 -1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ6(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + (j * zg) + (k - 1);
}


//RIGHT_UP DIRECTION
// +1 +1  0
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ7(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j + 1) * zg) + k;
}


//LEFT_DOWN DIRECTION
// -1 -1  0
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ8(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j - 1) * zg) + k;
}


//RIGHT_BACKWARD DIRECTION
// +1  0 +1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ9(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + (j * zg) + (k + 1);
}


//LEFT_FORWARD DIRECTION
// -1  0 -1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ10(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + (j * zg) + (k - 1);
}


//UP_BACKWARD DIRECTION
//  0 +1 +1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ11(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j + 1) * zg) + (k + 1);
}


//DOWN_FORWARD DIRECTION
//  0 -1 -1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ12(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j - 1) * zg) + (k - 1);
}


//RIGHT_DOWN DIRECTION
// +1 -1  0
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ13(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j - 1) * zg) + k;
}


//LEFT_UP DIRECTION
// -1 +1  0
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ14(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j + 1) * zg) + k;
}


//RIGHT_FORWARD DIRECTION
// +1  0 -1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ15(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + (j * zg) + (k - 1);
}


//LEFT_BACKWARD DIRECTION
// -1  0 +1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ16(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + (j * zg) + (k + 1);
}


//UP_FORWARD DIRECTION
//  0 +1 -1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ17(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j + 1) * zg) + (k - 1);
}


//DOWN_BACKWARD DIRECTION
//  0 -1 +1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ18(tNi i, tNi j, tNi k)
{
    return (i * yg * zg) + ((j - 1) * zg) + (k + 1);
}


//RIGHT_UP_BACKWARD DIRECTION
// +1 +1 +1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ19(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j + 1) * zg) + (k + 1);
}


//LEFT_DOWN_FORWARD DIRECTION
// -1 -1 -1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ20(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j - 1) * zg) + (k - 1);
}


//RIGHT_UP_FORWARD DIRECTION
// +1 +1 -1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ21(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j + 1) * zg) + (k - 1);
}


//LEFT_DOWN_BACKWARD DIRECTION
// -1 -1 +1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ22(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j - 1) * zg) + (k + 1);
}


//RIGHT_DOWN_BACKWARD DIRECTION
// +1 -1 +1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ23(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j - 1) * zg) + (k + 1);
}


//LEFT_UP_FORWARD DIRECTION
// -1 +1 -1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ24(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j + 1) * zg) + (k - 1);
}


//LEFT_UP_BACKWARD DIRECTION
// -1 +1 +1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ25(tNi i, tNi j, tNi k)
{
    return ((i - 1) * yg * zg) + ((j + 1) * zg) + (k + 1);
}


//RIGHT_DOWN_FORWARD DIRECTION
// +1 -1 -1
template <typename T, int QVecSize>
tNi inline ComputeUnit<T, QVecSize>::dirnQ26(tNi i, tNi j, tNi k)
{
    return ((i + 1) * yg * zg) + ((j - 1) * zg) + (k - 1);
}
