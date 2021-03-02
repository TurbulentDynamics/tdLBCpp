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

#include "../../tdLBCppApi/include/GlobalStructures.hpp"



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
};



template <typename T, int size=QLen::D3Q19>
struct QVec {
    T q[size];
    
    QVec() {
    }

    ~QVec() {
        freeMem();
    }
    QVec(const QVec<T, size>& other) {
        copy(other);
    }
    QVec<T, size>& operator=(const QVec<T, size>& other) {
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
    
	T& operator[](int i) {
		assert(i >=0 && i < size);
		return q[i];
	}

    
    
private:
    void freeMem() {
    }
    void copy(const QVec<T, size> &other) {
        if (this != &other) {
            freeMem();
            for (int l = 0; l < size; l++) {
                q[l] = other.q[l];
            }
        }
    }
};











