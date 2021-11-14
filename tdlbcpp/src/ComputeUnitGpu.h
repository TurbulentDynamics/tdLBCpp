#pragma once

#include "Header.h"

#include "ComputeUnit.h"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
class ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, GPU> : public ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>
{
public:
    using Base = ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, streamingType>;
    using Base::Base;
    using Base::flow; using Base::xg1; using Base::yg1; using Base::zg1;
    using Base::F; using Base::index; using Base::Q; using Base::Nu;
    using Base::size; using Base::O; using Base::deviceID; using Base::ExcludeOutputPoints;
    using Base::xg; using Base::yg; using Base::zg;
    using Base::init;

    Force<T> *devF;

    dim3 numBlocks;
    dim3 threadsPerBlock;
    ComputeUnitBase<T, QVecSize, MemoryLayout> *gpuThis;
    int gpuGeomSize;
    PosPolar<tNi, T> *gpuGeom;

    ComputeUnitArchitecture();
    ComputeUnitArchitecture(ComputeUnitParams cuJson, FlowParams<T> flow, DiskOutputTree outputTree);
    ComputeUnitArchitecture(ComputeUnitArchitecture &&) noexcept;
    void checkEnoughMemory();
    virtual void allocateMemory();
    virtual void freeMemory();
    virtual void architectureInit();
    virtual void initialise(T rho);
    virtual void forcing(std::vector<PosPolar<tNi, T>> &geom, T alfa, T beta);
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