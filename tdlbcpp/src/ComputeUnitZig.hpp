#pragma once

#include "ComputeUnit.h"
#include "ComputeUnitZig.h"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void collisionWrapper(ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG> *self);

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void collisionMomentsWrapper(ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG> *self);

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void bounceBackBoundaryWrapper(ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG> *self);

extern "C"
{
    void EgglesSomers_collision_zig__neive__float__long_int__19_ijkl(float *,
                                                                     float *,
                                                                     float *,
                                                                     long,
                                                                     long,
                                                                     long,
                                                                     float,
                                                                     float,
                                                                     float,
                                                                     int);

    void EgglesSomers_collision_zig__esotwist__float__long_int__19_ijkl(float *,
                                                                        float *,
                                                                        float *,
                                                                        long,
                                                                        long,
                                                                        long,
                                                                        float,
                                                                        float,
                                                                        float,
                                                                        int,
                                                                        int);
    void EgglesSomers_collision_moments_zig__neive__float__long_int__19_ijkl(float *,
                                                                             long,
                                                                             long,
                                                                             long);

    void EgglesSomers_collision_moments_zig__esotwist__float__long_int__19_ijkl(float *,
                                                                                long,
                                                                                long,
                                                                                long,
                                                                                int);
    void Bounce_zig__neive__float__long_int__19_ijkl(float *,
                                                     long,
                                                     long,
                                                     long);
}

template <>
void collisionWrapper(ComputeUnitArchitecture<float, D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple, CPU_ZIG> *self)
{
    EgglesSomers_collision_zig__neive__float__long_int__19_ijkl(self->Q.q,
                                                                (float *)self->F,
                                                                self->Nu,
                                                                self->xg,
                                                                self->yg,
                                                                self->zg,
                                                                self->flow.cs0,
                                                                self->flow.g3,
                                                                self->flow.nu,
                                                                (int)self->flow.useLES);
}

template <>
void collisionWrapper(ComputeUnitArchitecture<float, D3Q19, MemoryLayoutIJKL, EgglesSomers, Esotwist, CPU_ZIG> *self)
{
    EgglesSomers_collision_zig__esotwist__float__long_int__19_ijkl(self->Q.q,
                                                                   (float *)self->F,
                                                                   self->Nu,
                                                                   self->xg,
                                                                   self->yg,
                                                                   self->zg,
                                                                   self->flow.cs0,
                                                                   self->flow.g3,
                                                                   self->flow.nu,
                                                                   (int)self->flow.useLES,
                                                                   (int)self->evenStep);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG>::collision()
{
    ::collisionWrapper(this);
}

template <>
void collisionMomentsWrapper(ComputeUnitArchitecture<float, D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple, CPU_ZIG> *self)
{
    EgglesSomers_collision_moments_zig__neive__float__long_int__19_ijkl(self->Q.q,
                                                                        self->xg,
                                                                        self->yg,
                                                                        self->zg);
}

template <>
void collisionMomentsWrapper(ComputeUnitArchitecture<float, D3Q19, MemoryLayoutIJKL, EgglesSomers, Esotwist, CPU_ZIG> *self)
{
    EgglesSomers_collision_moments_zig__esotwist__float__long_int__19_ijkl(self->Q.q,
                                                                           self->xg,
                                                                           self->yg,
                                                                           self->zg,
                                                                           (int)self->evenStep);
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG>::moments()
{
    ::collisionMomentsWrapper(this);
}

template <>
void bounceBackBoundaryWrapper(ComputeUnitArchitecture<float, D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple, CPU_ZIG> *self)
{
    Bounce_zig__neive__float__long_int__19_ijkl(self->Q.q,
                                                self->xg,
                                                self->yg,
                                                self->zg);
}

template <>
void bounceBackBoundaryWrapper(ComputeUnitArchitecture<float, D3Q19, MemoryLayoutIJKL, EgglesSomers, Esotwist, CPU_ZIG> *self)
{
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG>::bounceBackBoundary()
{
    ::bounceBackBoundaryWrapper(this);
}