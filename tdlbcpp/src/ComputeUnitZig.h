#pragma once

#include "Header.h"

#include "ComputeUnit.h"

template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
class ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG>: public ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU> {
    using Base=ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU>;
    using Base::Base;

    virtual void collision();
    virtual void moments();
    virtual void bounceBackBoundary();
    virtual void forcing(std::vector<PosPolar<tNi, T>> &geom, T alfa, T beta);
};

#include "ComputeUnitZig.hpp"
