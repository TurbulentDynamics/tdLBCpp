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


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::initParams(ComputeUnitParams cuParams) {
    nodeID = cuParams.nodeID;
    deviceID = cuParams.deviceID;

    idi = cuParams.idi;
    idj = cuParams.idj;
    idk = cuParams.idk;

    resolution = cuParams.resolution;

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
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::allocateMemory() {
    Q.setSize(size);
    Q.q = new T[Q.qSize];
    F = new Force<T>[size];
    Nu = new T[size];
    O = new bool[size];
    ExcludeOutputPoints = new bool[size];
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::freeMemory() {
    if (Q.q != nullptr) {
        delete[] Q.q;
        Q.q = nullptr;
    }
    if (F != nullptr) {
        delete[] F;
        F = nullptr;
    }
    if (Nu != nullptr) {
        delete[] Nu;
        Nu = nullptr;
    }
    if (O != nullptr) {
        delete[] O;
        O = nullptr;
    }
    if (ExcludeOutputPoints != nullptr) {
        delete[] ExcludeOutputPoints;
        ExcludeOutputPoints = nullptr;
    }
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::architectureInit() {}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::init(ComputeUnitParams cuParams, bool reallocate) {

    initParams(cuParams);

    size_t new_size = size_t(xg) * yg * zg;
    if (reallocate && (new_size == size)) {
        reallocate = false;
    }
    size = new_size;

    if (reallocate) {
        freeMemory();
    }
    evenStep = true;

    allocateMemory();

    architectureInit();
}





template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
ComputeUnitBase<T, QVecSize, MemoryLayout>::ComputeUnitBase(ComputeUnitParams cuParams, FlowParams<T> flow, DiskOutputTree outputTree):flow(flow), outputTree(outputTree){
    init(cuParams, false);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
ComputeUnitBase<T, QVecSize, MemoryLayout>::ComputeUnitBase(ComputeUnitBase &&rhs) noexcept:
idi(rhs.idi), idj(rhs.idj), idk(rhs.idk), nodeID(rhs.nodeID), deviceID(rhs.deviceID),
x(rhs.x), y(rhs.y), z(rhs.z), i0(rhs.i0), j0(rhs.j0), k0(rhs.k0), xg(rhs.xg), yg(rhs.yg), zg(rhs.zg), xg0(rhs.xg0), yg0(rhs.yg0), zg0(rhs.zg0), xg1(rhs.xg1), yg1(rhs.yg1), zg1(rhs.zg1),
ghost(rhs.ghost), size(rhs.size), flow(rhs.flow), Q(std::move(rhs.Q)), F(rhs.F), Nu(rhs.Nu), O(rhs.O), ExcludeOutputPoints(rhs.ExcludeOutputPoints),
outputTree(rhs.outputTree), evenStep(rhs.evenStep)
{
    rhs.O = nullptr;
    rhs.Nu = nullptr;
    rhs.F = nullptr;
    rhs.ExcludeOutputPoints = nullptr;
}



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
ComputeUnitBase<T, QVecSize, MemoryLayout>::~ComputeUnitBase()
{
    freeMemory();
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

    std::stringstream text;
    text << " cu.fopen_read: Starting Checkpoint LOADING " << filePath << std::endl;
    outputTree.writeToRunningDataFileAndPrint(text.str());

    return fopen(filePath.c_str(), "r");
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
FILE* ComputeUnitBase<T, QVecSize, MemoryLayout>::fopen_write(std::string filePath){

    std::stringstream text;
    text << " cu.fopen_write: Open File for Writing:" << filePath << std::endl;
    outputTree.writeToRunningDataFileAndPrint(text.str());

    return fopen(filePath.c_str(), "w");
}





template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::doubleResolutionFullCU(){

    tNi factor = 2;

    resolution += factor;


    //Set up new pointers to copy from existing matrices
    Fld fromQ;
    fromQ.q = Q.q;
    Force<T> *fromF = F;
    T *fromNu = Nu;
    bool *fromO = O;
    bool *fromExcludeOutputPoints = ExcludeOutputPoints;



    


//    nodeID = cuParams.nodeID;
//    deviceID = cuParams.deviceID;
//
//    idi = cuParams.idi;
//    idj = cuParams.idj;
//    idk = cuParams.idk;

    x *= factor;
    y *= factor;
    z *= factor;

//    i0 = cuParams.i0;
//    j0 = cuParams.j0;
//    k0 = cuParams.k0;

//    ghost = cuParams.ghost;

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


    //Setup Size
    size = size_t(xg) * yg * zg;

    //Reallocate New Memory for Q, F, Nu, O, ExcludeOutputPoints
    allocateMemory();


    for (tNi i=0;  i <= xg0; i++) {
        for (tNi j=0;  j <= yg0; j++) {
            for (tNi k=0;  k <= zg0; k++) {

                Q[index(i,j,k)] = fromQ[indexIncreasingResolutionFROM(i,j,k)];

            }}}


    delete[] fromQ.q;
    delete[] fromF;
    delete[] fromNu;
    delete[] fromO;
    delete[] fromExcludeOutputPoints;


//Need to corordinate with creating NEW GPU MALLOC
//    architectureInit();


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



    std::string paramsFilePath = outputTree.getAllParamsFilePath(dirname, unit_name) + ".json";


    std::stringstream text;
    text << " Checkpoint Load: " << paramsFilePath << std::endl;


    outputTree.readAllParamsJson(paramsFilePath, binFormat, running);
    flow = outputTree.getFlowParams<T>();

    //TODO: Check this
    init(outputTree.getComputeUnitParams(), true);


    std::string filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "Q");
    FILE *fpQ = fopen_read(filePath);
    if (fpQ) {
        fread(Q, sizeof(QVec<T, QVecSize>), size, fpQ);
        fclose(fpQ);
    } else {
        std::stringstream text;
        // std::strerror(errno)
        text << " checkpoint_read: File couldn't be read: " << filePath << " "  << std::endl;
        outputTree.writeToRunningDataFileAndPrint(text.str());
    }


    filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "F");
    FILE *fpF = fopen_read(filePath);
    if (fpF) {
        fread(F, sizeof(Force<T>), size, fpF);
        fclose(fpF);
    } else {
        std::stringstream text;
        text << " checkpoint_read: File couldn't be read: " << filePath << " "  << std::endl;
        outputTree.writeToRunningDataFileAndPrint(text.str());
    }


    filePath = outputTree.getCheckpointFilePath(dirname, unit_name, "Nu");
    FILE *fpNu = fopen_read(filePath);
    if (fpNu) {
        fread(Nu, sizeof(T), size, fpNu);
        fclose(fpNu);
    } else {
        std::stringstream text;
        text << " checkpoint_read: File couldn't be read: " << filePath << " "  << std::endl;
        outputTree.writeToRunningDataFileAndPrint(text.str());

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


    std::stringstream text;
    text << " Starting Checkpoint Writing " << dirname;
    std::cout << text.str() << std::endl;
    outputTree.writeToRunningDataFile(text.str());


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


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
tNi inline ComputeUnitBase<T, QVecSize, MemoryLayout>::indexIncreasingResolutionFROM(tNi i, tNi j, tNi k)
{
    tNi factor = 2;

    tNi fromXG = x / factor + ghost * 2;
    tNi fromYG = y / factor + ghost * 2;
    tNi fromZG = z / factor + ghost * 2;

    tNi fromI = (i/factor + i % factor);
    tNi fromJ = (i/factor + i % factor);
    tNi fromK = (i/factor + i % factor);

#ifdef DEBUG

    if ((fromI>=fromXG) || (fromJ>=fromYG) || (fromK>=fromZG)) {
        std::cout << "Index Error  " << i <<" "<< xg <<" "<< j <<" "<< yg <<" "<< k <<" "<< zg << std::endl;
        exit(1);
    }
#endif
    return fromI * (fromYG * fromZG) + (fromJ * fromZG) + fromK;
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
