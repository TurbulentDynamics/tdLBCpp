//
//  QVec_hpp.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include <stdio.h>
#include <assert.h>

#include "Header.h"

#include "GlobalStructures.hpp"



template <typename T>
struct Velocity {
    T x, y, z;
    
    template <typename T2>
    Velocity<T2> returnAs(){
        Velocity<T2> u;
        u.x = T2(x);
        u.y = T2(y);
        u.z = T2(z);
        return u;
    }
};


template <typename T>
struct Force {
    T x, y, z;

    template <typename T2>
    Force<T2> returnAs(){
        Force<T2> f;
        f.x = T2(x);
        f.y = T2(y);
        f.z = T2(z);
        return f;
    }
    
    void setToZero(){
        x = 0.0;
        y = 0.0;
        z = 0.0;
        
    }
    
    bool isNotZero(){
        return (x!=0.0 && y!=0.0 && z!=0.0);
    }
};



template <typename T, int size=QLen::D3Q19>
struct QVecBase {
    T q[size];
    
    QVecBase() {
    }

    QVecBase(const T* qFrom) {
        copy(qFrom);
    }

    QVecBase(const T* qFrom, size_t step) {
        copy(qFrom, step);
    }

    ~QVecBase() {
        freeMem();
    }
    QVecBase(const QVecBase<T, size>& other) {
        copy(other);
    }
    QVecBase<T, size>& operator=(const QVecBase<T, size>& other) {
        copy(other);
        return *this;
    }
    
#ifndef RELEASE
	HOST_DEVICE_GPU T& operator[](int i) {
		assert(i >=0 && i < size);
		return q[i];
	}
#endif
    
    
private:
    void freeMem() {
    }

    inline void copy(const T* qFrom) {
        for (int l = 0; l < size; l++) {
            q[l] = qFrom[l];
        }
    }

    inline void copy(const T* qFrom, size_t step) {
        for (int l = 0; l < size; l++) {
            q[l] = *(qFrom + l * step);
        }
    }
    
    inline void copy(const QVecBase<T, size> &other) {
        if (this != &other) {
            freeMem();
            copy(other.q);
        }
    }
};

template<typename Base, typename T, int size>
struct CommonOperations : public Base {
    using Base::q;
    using Base::Base;
    using Base::operator=;

    Velocity<T> velocity(Force<T> f){


        Velocity<T> u;
        u.x = (1.0 / q[M01]) * (q[M02] + 0.5 * f.x);
        u.y = (1.0 / q[M01]) * (q[M03] + 0.5 * f.y);
        u.z = (1.0 / q[M01]) * (q[M04] + 0.5 * f.z);

        return u;
    }

    Velocity<T> velocity(){
        
        Velocity<T> u;
        u.x = q[M02] / q[M01];
        u.y = q[M03] / q[M01];
        u.z = q[M04] / q[M01];
      
        return u;
    };

    void initialiseRho(T initialRho, T other){
        q[MRHO] = initialRho;
        for (int l = 1; l < size; l++) {
            q[l] = other;
        }
    }

    void setToZero(){
        for (int l = 0; l < size; l++) {
            q[l] = 0.0;
        }
    }
};

template<typename T, int size=QLen::D3Q19>
using QVec=CommonOperations<QVecBase<T,size>,T,size>;









