//
//  ComputeUnit.h
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include <stdio.h>
#include <vector>

#ifndef DEBUG
#include <iostream>
#endif

#if WITH_GPU
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif


#include "Header.h"
#include "Params/Flow.hpp"
#include "Params/ComputeUnitParams.hpp"
#include "Params/BinFile.hpp"
#include "Params/OutputParams.hpp"

#include "Sources/tdLBGeometryRushtonTurbineLibCPP/RushtonTurbine.hpp"
#include "Sources/tdLBGeometryRushtonTurbineLibCPP/GeomPolar.hpp"

#include "Field.hpp"
#include "QVec.hpp"
#include "Params/Checkpoint.hpp"
#include "Params/Running.hpp"
#include "DiskOutputTree.h"
#include "Output.hpp"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
class ComputeUnitBase {
public:
    using Fld = Field<T, QVecSize, MemoryLayout>;
    
    //Position in the grid
    int idi, idj, idk;
    int nodeID;
    int deviceID;
    
    //Size of this ComputeUnit
    tNi x, y, z;
    
    //Starting absolute position in the grid
    tNi i0, j0, k0;

    
    
    tNi xg, yg, zg;
    tNi xg0, yg0, zg0;
    tNi xg1, yg1, zg1;
    
    
    tNi ghost;
    size_t size;
    
    
    FlowParams<T> flow;
    
    
    Fld Q;
    
    Force<T> * __restrict__ F;
    //    std::vector<Force<T>> sparseF;
        
    T * __restrict__ Nu;
    
    bool * __restrict__ O;

    bool * __restrict__ ExcludeOutputPoints;


    DiskOutputTree outputTree;
    bool evenStep;

    ComputeUnitBase();
    ComputeUnitBase(ComputeUnitParams cuJson, FlowParams<T> flow, DiskOutputTree outputTree);
    ComputeUnitBase(const ComputeUnitBase &) = delete;
    ComputeUnitBase(ComputeUnitBase &&) noexcept;
    ComputeUnitBase& operator=(const ComputeUnitBase &) = delete;
    ComputeUnitBase& operator=(ComputeUnitBase &&) noexcept = delete;
    
    ~ComputeUnitBase();
    
    void initParams(ComputeUnitParams);
    virtual void allocateMemory();
    virtual void freeMemory();
    virtual void architectureInit();
    void init(ComputeUnitParams, bool);
    HOST_DEVICE_GPU tNi inline index(tNi i, tNi j, tNi k);
    tNi inline indexPlusGhost(tNi i, tNi j, tNi k);


    Velocity<T> inline getVelocity(tNi i, tNi j, tNi k);
    Velocity<T> inline getVelocitySparseF(tNi i, tNi j, tNi k, Force<T> f);

    void setQToZero();
    virtual void initialise(T rho);


    virtual void forcing(std::vector<PosPolar<tNi, T>> &geom, T alfa, T beta) = 0;
    
    
    virtual void bounceBackBoundary() = 0;
    void bounceBackBoundaryRight();
    void bounceBackBoundaryLeft();
    void bounceBackBoundaryUp();
    void bounceBackBoundaryDown();
    void bounceBackBoundaryBackward();
    void bounceBackBoundaryForward();
    void bounceBackEdges();

    void setGhostSizes();
    void getParamsFromJson(const std::string filePath);
    int writeParamsToJsonFile(const std::string filePath);
    Json::Value getJson();
    void printParams();

        
    
    void checkpoint_read(std::string dirname, std::string unit_name);
    void checkpoint_write(std::string unit_name, RunningParams run);
    



    bool hasOutputAtStep(OutputParams output, RunningParams running);
    
    void setOutputExcludePoints(std::vector<Pos3d<tNi>> geomPoints);
    void setOutputExcludePoints(std::vector<PosPolar<tNi, T>> geomPoints);

    void unsetOutputExcludePoints(std::vector<Pos3d<tNi>> geomPoints);
    void unsetOutputExcludePoints(std::vector<PosPolar<tNi, T>> geomPoints);


    template <typename tDiskPrecision, int tDiskSize>
    void savePlaneXZ(OrthoPlane plane, BinFileParams binFormat, RunningParams runParam);

    void writeAllOutput(RushtonTurbinePolarCPP<tNi, T> geom, OutputParams output, BinFileParams binFormat, RunningParams runParam);

    //Debug
    virtual void calcVorticityXZ(tNi j, RunningParams runParam) = 0;
    virtual void calcVorticityXY(tNi k, RunningParams runParam) = 0;

    
    
    
private:

    FILE* fopen_read(std::string filePath);
    FILE* fopen_write(std::string filePath);

public:    
    
    HOST_DEVICE_GPU tNi inline dirnQ00(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ01(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ02(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ03(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ04(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ05(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ06(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ07(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ08(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ09(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ10(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ11(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ12(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ13(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ14(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ15(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ16(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ17(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ18(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ19(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ20(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ21(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ22(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ23(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ24(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ25(tNi i, tNi j, tNi k);
    HOST_DEVICE_GPU tNi inline dirnQ26(tNi i, tNi j, tNi k);

    virtual void collision() = 0;
    virtual void moments() = 0;
    virtual void streamingPull() = 0;
    virtual void streamingPush() = 0;
    virtual void printDebug(int fi, int ti, int fj, int tj, int fk, int tk) = 0;


};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
class ComputeUnitForcing : public ComputeUnitBase<T, QVecSize, MemoryLayout> {
public:
    using Base=ComputeUnitBase<T, QVecSize, MemoryLayout>;
    using Base::flow; using Base::xg1; using Base::yg1; using Base::zg1;
    using Base::ghost; using Base::xg0; using Base::yg0; using Base::zg0;
    using Base::F; using Base::index; using Base::Q; using Base::O;
    using Base::Base;
    using Base::evenStep;
    using Base::size;
    using Base::ExcludeOutputPoints;
    using Base::outputTree;
    virtual void moments();
    virtual void forcing(std::vector<PosPolar<tNi, T>> &geom, T alfa, T beta);
    virtual void calcVorticityXZ(tNi j, RunningParams runParam);
    virtual void calcVorticityXY(tNi k, RunningParams runParam);
    virtual void printDebug(int fi, int ti, int fj, int tj, int fk, int tk);
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
class ComputeUnitCollision : public ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, streamingType> {};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
class ComputeUnitCollision<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>: public ComputeUnitForcing<T, QVecSize, MemoryLayout, EgglesSomers, streamingType> {
public:
    using Base=ComputeUnitForcing<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>;
    using Base::flow; using Base::xg1; using Base::yg1; using Base::zg1;
    using Base::F; using Base::index; using Base::Q; using Base::Nu;
    using Base::Base;
    using Base::evenStep;
    virtual void collision();
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
class ComputeUnitCollision<T, QVecSize, MemoryLayout, EgglesSomersLES, streamingType>: public ComputeUnitBase<T, QVecSize, MemoryLayout> {
    virtual void collision();
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
class ComputeUnitCollision<T, QVecSize, MemoryLayout, Entropic, streamingType>: public ComputeUnitBase<T, QVecSize, MemoryLayout> {
    virtual void collision();
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
class ComputeUnitStreaming {};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
class ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Simple>: public ComputeUnitCollision<T, QVecSize, MemoryLayout, collisionType, Simple> {
public:
    using Base=ComputeUnitCollision<T, QVecSize, MemoryLayout, collisionType, Simple>;
    using Base::flow; using Base::xg1; using Base::yg1; using Base::zg1; using Base::index; using Base::Q;
    using Base::dirnQ01; using Base::dirnQ02; using Base::dirnQ03; using Base::dirnQ04; using Base::dirnQ05;
    using Base::dirnQ06; using Base::dirnQ07; using Base::dirnQ08; using Base::dirnQ09; using Base::dirnQ10;
    using Base::dirnQ11; using Base::dirnQ12; using Base::dirnQ13; using Base::dirnQ14; using Base::dirnQ15;
    using Base::dirnQ16; using Base::dirnQ17; using Base::dirnQ18;
    using Base::bounceBackBoundaryRight; using Base::bounceBackBoundaryLeft;
    using Base::bounceBackBoundaryUp; using Base::bounceBackBoundaryDown;
    using Base::bounceBackBoundaryForward; using Base::bounceBackBoundaryBackward;
    using Base::bounceBackEdges;
    using Base::Base;
    virtual void streamingPull();
    virtual void streamingPush();
    virtual void bounceBackBoundary();
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
class ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Esotwist>: public ComputeUnitCollision<T, QVecSize, MemoryLayout, collisionType, Esotwist> {
public:
    using Base=ComputeUnitCollision<T, QVecSize, MemoryLayout, collisionType, Esotwist>;
    using Base::Base;
    using Base::evenStep;
    virtual void streamingPush();
    virtual void streamingPull();
    virtual void bounceBackBoundary();
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType, Architecture cuArchitecture>
class ComputeUnitArchitecture {};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
class ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU>: public ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType> {
    using Base=ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>;
    using Base::Base;
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType, Architecture cuArchitecture>
using ComputeUnit=ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, cuArchitecture>;

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
struct AccessField {};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
struct AccessField<T, QVecSize, MemoryLayout, collisionType, Simple> {
    inline static HOST_DEVICE_GPU QVec<T, QVecSize> read(ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, Simple> &cu, tNi i, tNi j, tNi k) {
        return cu.Q[cu.index(i,j,k)];
    }
    inline static HOST_DEVICE_GPU void write(ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, Simple> &cu, QVec<T, QVecSize> &q, tNi i, tNi j, tNi k) {
        cu.Q[cu.index(i,j,k)] = q;
    }
    inline static HOST_DEVICE_GPU void writeMoments(ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, Simple> &cu, QVec<T, QVecSize> &q, tNi i, tNi j, tNi k) {
        write(cu, q, i, j, k);
    }
 };

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, streamingType>::printDebug(int fi, int ti, int fj, int tj, int fk, int tk) {
    using AF = AccessField<T, QVecSize, MemoryLayout, collisionType, streamingType>;

    for (int i = fi; i <= fi; i++) {
        for (int j = fj; j <= tj; j++) {
            for (int k = fk; k <= tk; k++) {
                QVec<T, QVecSize> q = AF::read(*this, i, j, k);
                printf("index(%d, %d, %d):\n", i, j, k);
                for (int l = 0; l < QVecSize; l++) {
                    printf("Q[%d] = %f, ", l, Q[index(i,j,k)][l]);
                }
                printf("\n read: ");
                for (int l = 0; l < QVecSize; l++) {
                    printf("q[%d] = %f, ", l, q[l]);
                }
                printf("\n");
            }
        }
    }
}

#include "ComputeUnit.hpp"
#include "ComputeUnitOutput.hpp"
#include "CollisionEgglesSomers.hpp"
#include "CollisionEntropic.hpp"
#include "StreamingNieve.hpp"
#include "StreamingEsoteric.hpp"
#include "Boundary.hpp"
#include "Forcing.hpp"

#if defined(WITH_GPU) || defined(WITH_GPU_MEMSHARED)
#include "ComputeUnitGpu.h"
#endif
