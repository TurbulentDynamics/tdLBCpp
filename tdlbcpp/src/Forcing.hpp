//
//  Forcing.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include <cmath>


#include "ComputeUnit.h"

#include "Sources/tdLBGeometryRushtonTurbineLibCPP/GeomPolar.hpp"





template <typename T>
T inline calcWeight(T x){
    
    T weight = 0.;
    
    x = fabs(x);    //NB abs returns an interger, fabs returns a float
    
    if (x <= 1.5) {
        if (x <= 0.5) {
            weight = (1.0 / 3.0) * (1.0 + sqrt(-3.0 * pow(x, 2) + 1.0));
        }else{
            weight = (1.0 / 6.0) * (5.0 - (3.0 * x) - sqrt((-3.0 * pow((1.0 - x), 2)) + 1.0));
        }
    }
    
    return weight;
}




template <typename T>
void inline smoothedDeltaFunction(T i_cart_fraction, T k_cart_fraction, T ppp[][3]){
    
    for (tNi k = -1; k <= 1; k++){
        for (tNi i = -1; i <= 1; i++){
            
            T hx = -i_cart_fraction + T(i);
            T hz = -k_cart_fraction + T(k);
            
            ppp[i+1][k+1] = calcWeight(hx) * calcWeight(hz);
            
        }
    }
}



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, streamingType>::forcing(std::vector<PosPolar<tNi, T>> geom, T alfa, T beta){
    using AF = AccessField<T, QVecSize, MemoryLayout, collisionType, streamingType>;

    for (tNi i=1; i<=xg1; i++) {
        for (tNi j = 1; j<=yg1; j++){
            for (tNi k = 1; k<=zg1; k++){
                
                O[index(i,j,k)] = 0;
            }
        }
    }
    
    
    
    
    for (auto &g: geom){
        
        T ppp[3][3];

        tNi i = g.i + ghost;
        tNi j = g.j + ghost;
        tNi k = g.k + ghost;

        
        smoothedDeltaFunction(g.iCartFraction, g.kCartFraction, ppp);
        
    
        T rhoSum = 0.0;
        T xSum = 0.0;
        T ySum = 0.0;
        T zSum = 0.0;
        
        
        
        for (tNi k1 = -1; k1<=1; k1++){
            for (tNi i1 = -1; i1<=1; i1++){
                
                tNi i2 = i + i1;
                tNi j2 = j;
                tNi k2 = k + k1;

                if (i2 == 0)   i2 = xg1;
                if (i2 == xg0) i2 = 1;
                if (k2 == 0)   k2 = zg1;
                if (k2 == zg0) k2 = 1;

                QVec<T, QVecSize> q = AF::read(*this, i2, j2, k2);
                T rho = q[M01];
                
                Force<T> localForce = F[index(i2,j2,k2)];

                T x = q[M02] + 0.5 * localForce.x;
                T y = q[M03] + 0.5 * localForce.y;
                T z = q[M04] + 0.5 * localForce.z;


                //adding the density of a nearby point using a weight (in ppp)
                rhoSum += ppp[i1+1][k1+1] * rho;

                //adding the velocity of a nearby point using a weight (in ppp)
                xSum += ppp[i1+1][k1+1] * x;
                ySum += ppp[i1+1][k1+1] * y;
                zSum += ppp[i1+1][k1+1] * z;
            }
        }//endfor  j1, k1
        
        
        //calculating the difference between the actual (weighted) speed and
        //the required (no-slip) velocity
        xSum -= rhoSum * g.uDelta;
        ySum -= rhoSum * g.vDelta;
        zSum -= rhoSum * g.wDelta;
        
        
        for (tNi k1 = -1; k1<=1; k1++){
            for (tNi i1 = -1; i1<=1; i1++){
                
                //printf("fx,  %i %i %i % 1.8E % 1.8E % 1.8E  \n", i2, j2, k2, xs, ys, zs);
                
                tNi i2 = i + i1;
                tNi j2 = j;
                tNi k2 = k + k1;


                if (i2 == 0)   i2 = xg1;
                if (i2 == xg0) i2 = 1;
                if (k2 == 0)   k2 = zg1;
                if (k2 == zg0) k2 = 1;
                
            

                Force<T> localForce = F[index(i2,j2,k2)];

                F[index(i2,j2,k2)].x = alfa * localForce.x - beta * ppp[i1+1][k1+1] * xSum;
                F[index(i2,j2,k2)].y = alfa * localForce.y - beta * ppp[i1+1][k1+1] * ySum;
                F[index(i2,j2,k2)].z = alfa * localForce.z - beta * ppp[i1+1][k1+1] * zSum;
                
                
                O[index(i2,j2,k2)] = 1;
                
            }}//endfor  j1, k1
        
    }//endfor
    
    
    

    for (tNi i=1; i<=xg1; i++) {
        for (tNi j = 1; j<=yg1; j++){
            for (tNi k = 1; k<=zg1; k++){
                
                //pos =  offsetb + offsets + k;
                if (O[index(i,j,k)] == 0) {
                    F[index(i,j,k)].x = 0.0;
                    F[index(i,j,k)].y = 0.0;
                    F[index(i,j,k)].z = 0.0;
                } else {
                    //Set it back to 0
                    O[index(i,j,k)] = 0;
                }//endif
            }}}//endfor  ijk
    
}//end of func





