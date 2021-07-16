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
    };
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
    };
    
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
    
    void setToZero(){
        for (int l = 0; l < size; l++) {
            q[l] = 0.0;
        }
    };
    
    Velocity<T> velocity(Force<T> f){
        
        Velocity<T> u;
        u.x = (1.0 / q[ Q1]) * (q[ Q2] + 0.5 * f.x);
        u.y = (1.0 / q[ Q1]) * (q[ Q3] + 0.5 * f.y);
        u.z = (1.0 / q[ Q1]) * (q[ Q4] + 0.5 * f.z);

        return u;
    };
    
    Velocity<T> velocity(){
        
        Velocity<T> u;
        u.x = q[Q2] / q[Q1];
        u.y = q[Q3] / q[Q1];
        u.z = q[Q4] / q[Q1];
      
        return u;
    };
    
#ifndef RELEASE
	T& operator[](int i) {
		assert(i >=0 && i < size);
		return q[i];
	}
#endif
    
    
private:
    void freeMem() {
    }
    
    void copy(const QVecBase<T, size> &other) {
        if (this != &other) {
            freeMem();
            for (int l = 0; l < size; l++) {
                q[l] = other.q[l];
            }
        }
    }
};

template <typename T, int size=QLen::D3Q19>
struct QVec {};

template <typename T>
struct QVec<T,QLen::D3Q27>:public QVecBase<T,QLen::D3Q27> {

    using QVecBase<T, QLen::D3Q27>::velocity;
    
    Velocity<T> velocity(T rho){
        
        Velocity<T> u;
        T *q = this->q;

        u.x = (q[Q1] - q[Q2] + q[Q7] - q[Q8] + q[Q9] - q[Q10] + q[Q13] - q[Q14] + q[Q15] - q[Q16]
            + q[Q19] - q[Q20] + q[Q21] - q[Q22] + q[Q23] - q[Q24] + q[Q26] - q[Q25]) / rho;
        u.y = (q[Q3] - q[Q4] + q[Q7] - q[Q8] + q[Q11] - q[Q12] + q[Q14] - q[Q13] + q[Q17] - q[Q18]
            + q[Q19] - q[Q20] + q[Q21] - q[Q22] + q[Q24] - q[Q23] + q[Q25] - q[Q26]) / rho;
        u.z = (q[Q5] - q[Q6] + q[Q9] - q[Q10] + q[Q11] - q[Q12] + q[Q16] - q[Q15] + q[Q18] - q[Q17]
            + q[Q19] - q[Q20] + q[Q22] - q[Q21] + q[Q23] - q[Q24] + q[Q25] - q[Q26]) / rho;
      
        return u;
    };
};

template <typename T>
struct QVec<T,QLen::D3Q19>:public QVecBase<T,QLen::D3Q19> {

    using QVecBase<T, QLen::D3Q19>::velocity;

    Velocity<T> velocity(T rho){
        
        Velocity<T> u;
        T *q = this->q;

        u.x = (q[Q1] - q[Q2] + q[Q7] - q[Q8] + q[Q9] - q[Q10] + q[Q13] - q[Q14] + q[Q15] - q[Q16]) / rho;
        u.y = (q[Q3] - q[Q4] + q[Q7] - q[Q8] + q[Q11] - q[Q12] + q[Q14] - q[Q13] + q[Q17] - q[Q18]) / rho;
        u.z = (q[Q5] - q[Q6] + q[Q9] - q[Q10] + q[Q11] - q[Q12] + q[Q16] - q[Q15] + q[Q18] - q[Q17]) / rho;
      
        return u;
    };
};









