#pragma once

#include "ComputeUnit.h"
#include "ComputeUnitZig.h"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void collisionWrapper(ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG> *self);

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void collisionMomentsWrapper(ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG> *self);

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void bounceBackBoundaryWrapper(ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG> *self);

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void forcingWrapper(ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG> *self, std::vector<PosPolar<tNi, T>> &geom, T alfa, T beta);

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
    void forcing_zig__neive__float__long_int__19_ijkl(float *,
                                                      float *,
                                                      float *,
                                                      bool *,
                                                      long,
                                                      long,
                                                      long,
                                                      long,
                                                      float,
                                                      float,
                                                      float,
                                                      int,
                                                      PosPolar<long, float> *gs,
                                                      int gsSize,
                                                      float alfa,
                                                      float beta);

    void forcing_zig__esotwist__float__long_int__19_ijkl(float *,
                                                         float *,
                                                         float *,
                                                         bool *,
                                                         long,
                                                         long,
                                                         long,
                                                         long,
                                                         float,
                                                         float,
                                                         float,
                                                         int,
                                                         int,
                                                         PosPolar<long, float> *gs,
                                                         int gsSize,
                                                         float alfa,
                                                         float beta);
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

template <>
void forcingWrapper(ComputeUnitArchitecture<float, D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple, CPU_ZIG> *self, std::vector<PosPolar<long, float>> &geom, float alfa, float beta)
{
    forcing_zig__neive__float__long_int__19_ijkl(self->Q.q,
                                                 (float *)self->F,
                                                 self->Nu,
                                                 self->O,
                                                 self->xg,
                                                 self->yg,
                                                 self->zg,
                                                 self->ghost,
                                                 self->flow.cs0,
                                                 self->flow.g3,
                                                 self->flow.nu,
                                                 (int)self->flow.useLES,
                                                 geom.data(),
                                                 geom.size(),
                                                 alfa,
                                                 beta);
}

template <>
void forcingWrapper(ComputeUnitArchitecture<float, D3Q19, MemoryLayoutIJKL, EgglesSomers, Esotwist, CPU_ZIG> *self, std::vector<PosPolar<long, float>> &geom, float alfa, float beta)
{
    forcing_zig__esotwist__float__long_int__19_ijkl(self->Q.q,
                                                    (float *)self->F,
                                                    self->Nu,
                                                    self->O,
                                                    self->xg,
                                                    self->yg,
                                                    self->zg,
                                                    self->ghost,
                                                    self->flow.cs0,
                                                    self->flow.g3,
                                                    self->flow.nu,
                                                    (int)self->flow.useLES,
                                                    (int)self->evenStep,
                                                    geom.data(),
                                                    geom.size(),
                                                    alfa,
                                                    beta);
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitArchitecture<T, QVecSize, MemoryLayout, collisionType, streamingType, CPU_ZIG>::forcing(std::vector<PosPolar<tNi, T>> &geom, T alfa, T beta)
{
    ::forcingWrapper(this, geom, alfa, beta);
}
