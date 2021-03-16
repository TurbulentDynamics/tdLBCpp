//
//  tdLBCppApi/GlobalStructures.hpp
//
//
//  Created by Niall Ó Broin on 16/02/2021.
//

#ifndef Header_GlobalStructures
#define Header_GlobalStructures




#include <stdio.h>



template <typename T>
struct Pos2d{
    T p;
    T q;
};



template <typename T>
struct Pos3d{
    T i;
    T j;
    T k;
};


template <typename T>
struct Line2d{
    T x0;
    T y0;
    T x1;
    T y1;
};



template <typename T>
struct Extents{
    T x0;
    T x1;
    T y0;
    T y1;
    T z0;
    T z1;

    Extents() {
    }

    Extents(
            T x0,
            T x1,
            T y0,
            T y1,
            T z0,
            T z1
            ) {

        this-> x0 = x0;
        this-> x1 = x1;
        this-> y0 = y0;
        this-> y1 = y1;
        this-> z0 = z0;
        this-> z1 = z1;
    }

    bool containsI(T i){
        if (i >= x0 && i <= x1) return true;
        else return false;
    }
    bool containsJ(T j){
        if (j > y0 && j < y1) return true;
        else return false;
    }
    bool containsK(T k){
        if (k > z0 && k < z1) return true;
        else return false;
    }
    bool containsIK(T i, T k){
        if (i > x0 && i < x1 && k > z0 && k < z1) return true;
        else return false;
    }


    bool doesntContainI(T i){
        if (i < x0 || i > x1) return true;
        else return false;
    }
    bool doesntContainJ(T j){
        if (j < y0 || j > y1) return true;
        else return false;
    }
    bool doesntContainK(T k){
        if (k < z0 || k > z1) return true;
        else return false;
    }
    bool doesntContainIK(T i, T k){
        if (i < x0 || i > x1 || k < z0 || k > z1) return true;
        else return false;
    }
    bool doesntContainIJK(T i, T j, T k){
        if (i < x0 || i > x1 || j < y0 || j > y1 || k < z0 || k > z1) return true;
        else return false;
    }
};


#endif /* Header_h */
