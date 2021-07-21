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



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::forcing(std::vector<PosPolar<tNi, T>> geom, T alfa, T beta, tNi iCenter, tNi kCenter, tNi radius){
    
    
    //        T alfa = 0.97;
    //        T beta = 1.9;
    
    for (tNi i=1; i<=xg1; i++) {
        for (tNi j = 1; j<=yg1; j++){
            for (tNi k = 1; k<=zg1; k++){
                
                O[index(i,j,k)] = 0;
            }
        }
    }
    
    
    
    
    for (auto &g: geom){
        
        T ppp[3][3];
        
        
        T j_cart_fraction = 0;
        T k_cart_fraction = 0;
        
//        Pos3d<T> fraction = rotationForce<T>(iCenter, kCenter, radius, g);

//        smoothedDeltaFunction(fraction.i, fraction.k, ppp);
        
        
        tNi i = g.i;
        tNi j = g.j;
        tNi k = g.k;
    
        T rhos = 0.0;
        T xs = 0.0;
        T ys = 0.0;
        T zs = 0.0;
        
        
        
        for (tNi k1 = -1; k1<=1; k1++){
            for (tNi i1 = -1; i1<=1; i1++){
                
                tNi i2 = i + i1;
                tNi j2 = j;
                tNi k2 = k + k1;
                
//                if (j2 < 1)  j2 = j2 + ny;
//                if (j2 > ny) j2 = j2 - ny;
//                if (k2 < 1)  k2 = k2 + nz;
//                if (k2 > nz) k2 = k2 - nz;
                
                
                T rho = Q[index(i2,j2,k2)].q[RHOQ];
                
                
                Velocity<T> u = Q[index(i2,j2,k2)].velocity();
                
                
                //adding the density of a nearby point using a weight (in ppp)
                rhos += ppp[i1+1][k1+1] * rho;
                
                //printf("  %f %f   %f  \n", rho, rhos, ppp[j1+1][k1+1] );
                
                
                //adding the velocity of a nearby point using a weight (in ppp)
                xs += ppp[i1+1][k1+1] * u.x;
                ys += ppp[i1+1][k1+1] * u.y;
                zs += ppp[i1+1][k1+1] * u.z;
            }
        }//endfor  j1, k1
        
        
        //calculating the difference between the actual (weighted) speed and
        //the required (no-slip) velocity
        xs -= rhos * g.uDelta;
        ys -= rhos * g.vDelta;
        zs -= rhos * g.wDelta;
        
        
        for (tNi k1 = -1; k1<=1; k1++){
            for (tNi i1 = -1; i1<=1; i1++){
                
                //printf("fx,  %i %i %i % 1.8E % 1.8E % 1.8E  \n", i2, j2, k2, xs, ys, zs);
                
                tNi i2 = i + i1;
                tNi j2 = j;
                tNi k2 = k + k1;

//                if (j2 < 1)  j2 = j2+ny;
//                if (j2 > ny) j2 = j2-ny;
//                if (k2 < 1)  k2 = k2+nz;
//                if (k2 > nz) k2 = k2-nz;
                
            
                Velocity<T> u = Q[index(i2,j2,k2)].velocity();

                F[index(i2,j2,k2)].x = alfa * u.x - beta * ppp[i1+1][k1+1] * xs;
                F[index(i2,j2,k2)].y = alfa * u.y - beta * ppp[i1+1][k1+1] * ys;
                F[index(i2,j2,k2)].z = alfa * u.z - beta * ppp[i1+1][k1+1] * zs;
                
                
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





