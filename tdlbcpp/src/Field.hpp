
#include "Header.h"
#include "QVec.hpp"

template <typename T, int QVecSize>
struct QVecAccessCommon
{
    T *q;
    QVecAccessCommon(T *q) : q(q) {}
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
struct QVecAccessBase
{
};

template <typename T, int QVecSize>
struct QVecAccessBase<T, QVecSize, MemoryLayoutIJKL> : public QVecAccessCommon<T, QVecSize>
{
    using Base = QVecAccessCommon<T, QVecSize>;
    using Base::q;

    QVecAccessBase(T *Q, tNi index) : Base(Q + index * QVecSize) {}

    T &operator[](int l)
    {
        return q[l];
    }

    operator QVec<T, QVecSize>()
    {
        return QVec<T, QVecSize>(q);
    }

    QVecAccessBase<T, QVecSize, MemoryLayoutIJKL> &operator=(const QVec<T, QVecSize> &v)
    {
        for (int i = 0; i < QVecSize; i++)
        {
            q[i] = v.q[i];
        }
        return *this;
    }
};

template <typename T, int QVecSize>
struct QVecAccessBase<T, QVecSize, MemoryLayoutLIJK> : public QVecAccessCommon<T, QVecSize>
{
    using Base = QVecAccessCommon<T, QVecSize>;
    using Base::operator=;
    using Base::q;

    tNi ijkSize;

    QVecAccessBase(T *Q, tNi index, tNi ijkSize) : Base(Q + index), ijkSize(ijkSize) {}

    T &operator[](int l)
    {
        return q[l * ijkSize];
    }

    operator QVec<T, QVecSize>()
    {
        return QVec<T, QVecSize>(q, ijkSize);
    }

    QVecAccessBase<T, QVecSize, MemoryLayoutLIJK> &operator=(const QVec<T, QVecSize> &v)
    {
        for (int i = 0; i < QVecSize; i++)
        {
            (*this)[i] = v.q[i];
        }
        return *this;
    }
};

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
using QVecAccess = VelocityCalculation<QVecAccessBase<T, QVecSize, MemoryLayout>, T>;

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
struct FieldBase
{
    using QVecAcc = QVecAccess<T, QVecSize, MemoryLayout>;

    T *q;
    size_t qVectorNumber;
    size_t qSize;

    FieldBase()
    {
        q = 0;
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
    }

    QVecAcc operator[](int index)
    {
        return QVecAcc(q, index);
    }

    operator void *()
    {
        return q;
    }

    operator QVec<T, QVecSize>()
    {
    }

    ~FieldBase()
    {
        if (q != 0)
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

    QVecAcc operator[](int index)
    {
        return QVecAcc(q, index, qVectorNumber);
    }
};