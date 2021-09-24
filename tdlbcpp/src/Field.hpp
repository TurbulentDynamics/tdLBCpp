#if WITH_GPU
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#endif

#include "Header.h"
#include "QVec.hpp"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
struct QVecAccessBase
{
};

template <typename T, int QVecSize>
struct QVecAccessBase<T, QVecSize, MemoryLayoutIJKL>
{
    T *q;

    QVecAccessBase(T *Q, tNi index) : q(Q + index * QVecSize) {}

    inline T &operator[](tNi l)
    {
        return q[l];
    }

    inline operator QVec<T, QVecSize>()
    {
        return QVec<T, QVecSize>(q);
    }

    inline QVecAccessBase<T, QVecSize, MemoryLayoutIJKL> &operator=(const QVec<T, QVecSize> &v)
    {
        for (int i = 0; i < QVecSize; i++)
        {
            q[i] = v.q[i];
        }
        return *this;
    }
};

template <typename T>
struct SparseArray
{
    T *q;
    size_t step;
    SparseArray(T *q, size_t step) : q(q), step(step) {}
    inline T &operator[](tNi l)
    {
        return q[l * step];
    }
};

template <typename T, int QVecSize>
struct QVecAccessBase<T, QVecSize, MemoryLayoutLIJK>
{
    SparseArray<T> q;

    tNi ijkSize;

    QVecAccessBase(T *Q, tNi index, tNi ijkSize) : q(Q + index, ijkSize), ijkSize(ijkSize) {}

    inline T &operator[](tNi l)
    {
        return q[l];
    }

    inline operator QVec<T, QVecSize>()
    {
        return QVec<T, QVecSize>(q.q, ijkSize);
    }

    inline QVecAccessBase<T, QVecSize, MemoryLayoutLIJK> &operator=(const QVec<T, QVecSize> &v)
    {
        for (int i = 0; i < QVecSize; i++)
        {
            (*this)[i] = v.q[i];
        }
        return *this;
    }
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
using QVecAccess = CommonOperations<QVecAccessBase<T, QVecSize, MemoryLayout>, T, QVecSize>;

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
struct FieldBase
{
    using QVecAcc = QVecAccess<T, QVecSize, MemoryLayout>;

    bool allocated;
    T * __restrict__ q;
    size_t qVectorNumber;
    size_t qSize;

    FieldBase()
    {
        q = 0;
    }

    FieldBase(FieldBase&) = delete;
    FieldBase(FieldBase&& rhs) noexcept : allocated(false), q(rhs.q), qVectorNumber(rhs.qVectorNumber), qSize(rhs.qSize) {
        rhs.q = nullptr;
    }

    void allocate(size_t vectorNumber)
    {
        if (q != 0)
        {
            delete[] q;
        }
        qVectorNumber = vectorNumber;
        qSize = vectorNumber * QVecSize;
        q = new T[qSize];
        allocated = true;

#if WITH_GPU
        checkCudaErrors(cudaMalloc((void **)&q, sizeof(T) * qSize);
#endif
    }

    inline QVecAcc operator[](tNi index)
    {
        return QVecAcc(q, index);
    }

    inline operator void *()
    {
        return q;
    }

    ~FieldBase()
    {
        if (allocated && (q != 0))
        {
            delete[] q;
            q = 0;
        }
    }
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
struct Field : public FieldBase<T, QVecSize, MemoryLayout>
{
};

template <typename T, int QVecSize>
struct Field<T, QVecSize, MemoryLayoutLIJK> : public FieldBase<T, QVecSize, MemoryLayoutLIJK>
{
    using Base = FieldBase<T, QVecSize, MemoryLayoutLIJK>;
    using QVecAcc = QVecAccess<T, QVecSize, MemoryLayoutLIJK>;

    using Base::q;
    using Base::qVectorNumber;

    using Base::Base;

    inline QVecAcc operator[](tNi index)
    {
        return QVecAcc(q, index, qVectorNumber);
    }
};
