//
//  ComputeUnit.h
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include <stdio.h>
#include <vector>

#ifdef DEBUG
#include <iostream>
#endif

#if WITH_GPU == 1
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif


#include "Header.h"
#include "Params/Flow.hpp"
#include "Params/ComputeUnitParams.hpp"
#include "Params/BinFile.hpp"
#include "Params/OutputParams.hpp"


#include "Field.hpp"
#include "QVec.hpp"
#include "Params/Checkpoint.hpp"
#include "Params/Running.hpp"
#include "DiskOutputTree.h"
#include "Output.hpp"

#include "Sources/tdLBGeometryRushtonTurbineLibCPP/RushtonTurbine.hpp"
#include "Sources/tdLBGeometryRushtonTurbineLibCPP/GeomPolar.hpp"











template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
class ComputeUnitBase {
    using Fld = Field<T, QVecSize, MemoryLayout>;
public:
    
    //Position in the grid
    int idi, idj, idk;
    int mpiRank;

    
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
    
    Force<T> *F;
    //    std::vector<Force<T>> sparseF;
        
    T *Nu;
    
    bool *O;

    bool *excludeGeomPoints;
#if WITH_GPU == 1
    Force<T> *devF;
    T *devNu;
    void *devN;

    dim3 threadsPerBlock;
    dim3 numBlocks;
#endif

    
    DiskOutputTree outputTree;

    ComputeUnitBase();
    ComputeUnitBase(ComputeUnitParams cuJson, FlowParams<T> flow, DiskOutputTree outputTree);
    ComputeUnitBase(const ComputeUnitBase &) = delete;
    ComputeUnitBase(ComputeUnitBase &&) noexcept;
    ComputeUnitBase& operator=(const ComputeUnitBase &) = delete;
    ComputeUnitBase& operator=(ComputeUnitBase &&) noexcept = delete;
    
    ~ComputeUnitBase();
    
    void init(ComputeUnitParams, bool);    
    tNi inline index(tNi i, tNi j, tNi k);
    tNi inline index(int i, int j, int k);

    Velocity<T> inline getVelocity(tNi i, tNi j, tNi k);
    Velocity<T> inline getVelocitySparseF(tNi i, tNi j, tNi k, Force<T> f);

    void setQToZero();
    void initialise(T rho);




    void forcing(std::vector<PosPolar<tNi, T>>, T alfa, T beta, tNi iCenter, tNi kCenter, tNi radius);
    
    
    void bounceBackBoundary();
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
    void initialiseExcludePoints(RushtonTurbinePolarCPP<tNi, T> geom);
    void writeAllOutput(RushtonTurbinePolarCPP<tNi, T> geom, OutputParams output, BinFileParams binFormat, RunningParams runParam);

    //Debug
    void calcVorticityXZ(tNi j, RunningParams runParam);

    
    
    
private:

    FILE* fopen_read(std::string filePath);
    FILE* fopen_write(std::string filePath);

public:    
    
    tNi inline dirnQ000(tNi i, tNi j, tNi k);
    tNi inline dirnQ1(tNi i, tNi j, tNi k);
    tNi inline dirnQ2(tNi i, tNi j, tNi k);
    tNi inline dirnQ3(tNi i, tNi j, tNi k);
    tNi inline dirnQ4(tNi i, tNi j, tNi k);
    tNi inline dirnQ5(tNi i, tNi j, tNi k);
    tNi inline dirnQ6(tNi i, tNi j, tNi k);
    tNi inline dirnQ7(tNi i, tNi j, tNi k);
    tNi inline dirnQ8(tNi i, tNi j, tNi k);
    tNi inline dirnQ9(tNi i, tNi j, tNi k);
    tNi inline dirnQ10(tNi i, tNi j, tNi k);
    tNi inline dirnQ11(tNi i, tNi j, tNi k);
    tNi inline dirnQ12(tNi i, tNi j, tNi k);
    tNi inline dirnQ13(tNi i, tNi j, tNi k);
    tNi inline dirnQ14(tNi i, tNi j, tNi k);
    tNi inline dirnQ15(tNi i, tNi j, tNi k);
    tNi inline dirnQ16(tNi i, tNi j, tNi k);
    tNi inline dirnQ17(tNi i, tNi j, tNi k);
    tNi inline dirnQ18(tNi i, tNi j, tNi k);
    tNi inline dirnQ19(tNi i, tNi j, tNi k);
    tNi inline dirnQ20(tNi i, tNi j, tNi k);
    tNi inline dirnQ21(tNi i, tNi j, tNi k);
    tNi inline dirnQ22(tNi i, tNi j, tNi k);
    tNi inline dirnQ23(tNi i, tNi j, tNi k);
    tNi inline dirnQ24(tNi i, tNi j, tNi k);
    tNi inline dirnQ25(tNi i, tNi j, tNi k);
    tNi inline dirnQ26(tNi i, tNi j, tNi k);
    
    
    
    
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
class ComputeUnitCollision {};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
class ComputeUnitCollision<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>: public ComputeUnitBase<T, QVecSize, MemoryLayout> {
public:
    using Base=ComputeUnitBase<T, QVecSize, MemoryLayout>;
    using Base::flow; using Base::xg1; using Base::yg1; using Base::zg1;
    using Base::F; using Base::index; using Base::Q;
    using Base::Base;
    void collision();
    void moments();
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
class ComputeUnitCollision<T, QVecSize, MemoryLayout, EgglesSomersLES, streamingType>: public ComputeUnitBase<T, QVecSize, MemoryLayout> {
    void collision();
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
class ComputeUnitCollision<T, QVecSize, MemoryLayout, Entropic, streamingType>: public ComputeUnitBase<T, QVecSize, MemoryLayout> {
    void collision();
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
class ComputeUnitStreaming {};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
class ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Simple>: public ComputeUnitCollision<T, QVecSize, MemoryLayout, collisionType, Simple> {
public:
    using Base=ComputeUnitCollision<T, QVecSize, MemoryLayout, collisionType, Simple>;
    using Base::flow; using Base::xg1; using Base::yg1; using Base::zg1; using Base::index; using Base::Q;
    using Base::dirnQ1; using Base::dirnQ2; using Base::dirnQ3; using Base::dirnQ4; using Base::dirnQ5;
    using Base::dirnQ6; using Base::dirnQ7; using Base::dirnQ8; using Base::dirnQ9; using Base::dirnQ10;
    using Base::dirnQ11; using Base::dirnQ12; using Base::dirnQ13; using Base::dirnQ14; using Base::dirnQ15;
    using Base::dirnQ16; using Base::dirnQ17; using Base::dirnQ18;
    using Base::Base;
    void streaming();
    void streaming2();
    void streamingDEBUG();
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
class ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Esotwist>: public ComputeUnitCollision<T, QVecSize, MemoryLayout, collisionType, Esotwist> {
public:
    using Base=ComputeUnitBase<T, QVecSize, MemoryLayout>;
    bool evenStep;
    void streaming();
};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
using ComputeUnit=ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>;

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
struct AccessField {};

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout>
struct AccessField<T, QVecSize, MemoryLayout, Simple> {
    inline static QVec<T, QVecSize> read(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, tNi i, tNi j, tNi k) {
        return cu.Q[cu.index(i,j,k)];
    }
    inline static void write(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, QVec<T, QVecSize> &q, tNi i, tNi j, tNi k) {
        cu.Q[cu.index(i,j,k)] = q;
    }
};

#include "ComputeUnit.hpp"
#include "ComputeUnitOutput.hpp"
#include "CollisionEgglesSomers.hpp"
#include "CollisionEntropic.hpp"
#include "StreamingNieve.hpp"
#include "StreamingEsoteric.hpp"
#include "Boundary.hpp"
#include "Forcing.hpp"

