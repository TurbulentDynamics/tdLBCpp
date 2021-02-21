//
//  ComputeGroup.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include <cmath>


#include "ComputeUnit.h"

//struct geomFracion

//template <typename T>
//Pos3d<T> rotationForce(const tNi iCenter, const tNi kCenter, const tNi radius, Pos3d<tNi> p){
//
//
//    //Angular velocity
//
//    double theta = 0.0;
//
//    double iFP = iCenter + radius * cosf(theta);
//    double kFP = kCenter + radius * sinf(theta);
//
//    double integerPart;
//
//    T iFrac = T(modf(iFP, &integerPart));
//    T kFrac = T(modf(kFP, &integerPart));
//
//    return Pos3d<T>(iFrac, 0.0, kFrac);
//
//
//    g.r_polar = r;
//    g.t_polar = theta;
//    g.i_cart_fp = center.x + r * cosf(theta);
//    g.j_cart_fp = (tGeomShape)y - 0.5f;
//    g.k_cart_fp = center.z + r * sinf(theta);
//
//    g.u_delta_fp = -wa * g.r_polar * sinf(g.t_polar);
//    g.v_delta_fp = 0.;
//    g.w_delta_fp = wa * g.r_polar * cosf(g.t_polar);
//}







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



template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::forcing(std::vector<Pos3d<tNi>> geom, T alfa, T beta, tNi iCenter, tNi kCenter, tNi radius){
    
    
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
            for (tNi j1 = -1; j1<=1; j1++){
                
                tNi i2 = i;
                tNi j2 = j + j1;
                tNi k2 = k + k1;
                
//                if (j2 < 1)  j2 = j2 + ny;
//                if (j2 > ny) j2 = j2 - ny;
//                if (k2 < 1)  k2 = k2 + nz;
//                if (k2 > nz) k2 = k2 - nz;
                
                
                T rho = Q[index(i2,j2,k2)].q[RHOQ];
                
                
                Velocity<T> u = Q[index(i2,j2,k2)].velocity();
                
                
                //adding the density of a nearby point using a weight (in ppp)
                rhos += ppp[j1+1][k1+1] * rho;
                
                //printf("  %f %f   %f  \n", rho, rhos, ppp[j1+1][k1+1] );
                
                
                //adding the velocity of a nearby point using a weight (in ppp)
                xs += ppp[j1+1][k1+1] * u.x;
                ys += ppp[j1+1][k1+1] * u.y;
                zs += ppp[j1+1][k1+1] * u.z;
            }
        }//endfor  j1, k1
        
        
        //calculating the difference between the actual (weighted) speed and
        //the required (no-slip) velocity
//        xs -= rhos * g.u_delta_fp;
//        ys -= rhos * g.v_delta_fp;
//        zs -= rhos * g.w_delta_fp;
        
        
        for (tNi k1 = -1; k1<=1; k1++){
            for (tNi j1 = -1; j1<=1; j1++){
                
                //printf("fx,  %i %i %i % 1.8E % 1.8E % 1.8E  \n", i2, j2, k2, xs, ys, zs);
                
                tNi i2 = i;
                tNi j2 = j + j1;
                tNi k2 = k + k1;

//                if (j2 < 1)  j2 = j2+ny;
//                if (j2 > ny) j2 = j2-ny;
//                if (k2 < 1)  k2 = k2+nz;
//                if (k2 > nz) k2 = k2-nz;
                
            
                Velocity<T> u = Q[index(i2,j2,k2)].velocity();

                F[i2,j2,k2].x = alfa * u.x - beta * ppp[j1+1][k1+1] * xs;
                F[i2,j2,k2].y = alfa * u.y - beta * ppp[j1+1][k1+1] * ys;
                F[i2,j2,k2].z = alfa * u.z - beta * ppp[j1+1][k1+1] * zs;
                
                
                O[i2,j2,k2] = 1;
                
            }}//endfor  j1, k1
        
    }//endfor
    
    
    

    for (tNi i=0; i<=xg1; i++) {
        for (tNi j = 1; j<=yg1; j++){
            for (tNi k = 1; k<=zg1; k++){
                
                //pos =  offsetb + offsets + k;
                if (O[i,j,k] == 0) {
                    F[i,j,k].x = 0.0;
                    F[i,j,k].y = 0.0;
                    F[i,j,k].z = 0.0;
                } else {
                    //Set it back to 0
                    O[i,j,k] = 0;
                }//endif
            }}}//endfor  ijk
    
}//end of func





