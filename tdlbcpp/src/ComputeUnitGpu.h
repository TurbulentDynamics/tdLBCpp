#pragma once

#include "Header.h"

#include "ComputeUnit.h"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
class ComputeUnitArchitectureCommonGPU : public ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>
{
public:
    using Base = ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>;
    using Current = ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, collisionType, streamingType>;
    using Base::Base;
    using Base::flow; using Base::xg1; using Base::yg1; using Base::zg1;
    using Base::F; using Base::index; using Base::Q; using Base::Nu;
    using Base::size; using Base::O; using Base::deviceID; using Base::ExcludeOutputPoints;
    using Base::xg; using Base::yg; using Base::zg;
    using Base::init;
    using Base::saveJpeg;

    dim3 numBlocks;
    dim3 threadsPerBlock;
    Current *gpuThis;
    int gpuGeomSize;
    PosPolar<tNi, T> *gpuGeom;
    T *VortXZ;
    T *VortXY;
    bool *devExcludeOutputPoints;

    ComputeUnitArchitectureCommonGPU();
    ComputeUnitArchitectureCommonGPU(ComputeUnitParams cuJson, FlowParams<T> flow, DiskOutputTree outputTree);
    ComputeUnitArchitectureCommonGPU(ComputeUnitArchitectureCommonGPU &&) noexcept;
    void checkEnoughMemory();
    virtual void allocateMemory();
    virtual void freeMemory();
    virtual void architectureInit();
    virtual void initialise(T rho);
    virtual void doubleResolutionFullCU();
    virtual void forcing(std::vector<PosPolar<tNi, T>> &geom, T alfa, T beta);
    //virtual void collision();
    //virtual void moments();
    //virtual void bounceBackBoundary();
    //virtual void streamingPush();
    virtual void calcVorticityXZ(tNi j, RunningParams runParam, int jpegCompression);
    virtual void calcVorticityXY(tNi k, RunningParams runParam, int jpegCompression);

    virtual void setOutputExcludePoints(std::vector<Pos3d<tNi>> geomPoints);
    virtual void setOutputExcludePoints(std::vector<PosPolar<tNi, T>> geomPoints);

    virtual void unsetOutputExcludePoints(std::vector<Pos3d<tNi>> geomPoints);
    virtual void unsetOutputExcludePoints(std::vector<PosPolar<tNi, T>> geomPoints);

    void checkpoint_read(std::string dirname, std::string unit_name);

    ~ComputeUnitArchitectureCommonGPU();
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
class ComputeUnitArchitectureCollision{};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
class ComputeUnitArchitectureCollision<T, QVecSize, MemoryLayout, EgglesSomers, streamingType> : public ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>
{
public:
    using Base = ComputeUnitArchitectureCommonGPU<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>;
    using Current = ComputeUnitArchitectureCollision<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>;
    using Base::Base;
    using Base::numBlocks; using Base::threadsPerBlock; using Base::gpuThis;
    virtual void collision();
    virtual void moments();
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
class ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, Simple, GPU> : public ComputeUnitArchitectureCollision<T, QVecSize, MemoryLayout, collisionType, Simple>
{
public:
    using Base = ComputeUnitArchitectureCollision<T, QVecSize, MemoryLayout, collisionType, Simple>;
    using Current = ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, Simple, GPU>;
    using Base::Base;
    using Base::numBlocks; using Base::threadsPerBlock; using Base::gpuThis;
    virtual void bounceBackBoundary();
    virtual void streamingPush();
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
class ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, Esotwist, GPU> : public ComputeUnitArchitectureCollision<T, QVecSize, MemoryLayout, collisionType, Esotwist>
{
public:
    using Base = ComputeUnitArchitectureCollision<T, QVecSize, MemoryLayout, collisionType, Esotwist>;
    using Current = ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, Esotwist, GPU>;
    using Base::Base;
    using Base::gpuThis; using Base::evenStep;
    virtual void streamingPush();
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
class ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU_MEM_SHARED> : public ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>
{
public:
    using Base = ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU>;
    using Base::Base;
    using Base::checkEnoughMemory;
    using Base::F; using Base::devF; using Base::O; using Base::Q; using Base::Nu; using Base::ExcludeOutputPoints;
    using Base::size; using Base::gpuThis;
    using Base::init;

    virtual void allocateMemory();
    virtual void freeMemory();
};

#include "ComputeUnitGpu.hpp"
