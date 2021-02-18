//
//  ComputeGroup.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"

template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::streaming(Streaming scheme) {

    switch( scheme ) {
    case Streaming(Simple):
        streaming_simple(); break;
    case Streaming(Esotwist):
        streaming_esotwist(); break;
    }
}



template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::streaming_esotwist() {
	//TODO
}



template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::streaming_simple(){
    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){
                //DST    =  SRC
                Q[index(i, j, k)].q[O14] = Q[index(i+1, j,   k  )].q[O14];
                Q[index(i, j, k)].q[O16] = Q[index(i,   j+1, k  )].q[O16];
                Q[index(i, j, k)].q[O22] = Q[index(i,   j,   k+1)].q[O22];

                Q[index(i, j, k)].q[O11] = Q[index(i+1, j-1, k  )].q[O11];
                Q[index(i, j, k)].q[O17] = Q[index(i+1, j+1, k  )].q[O17];
                
                Q[index(i, j, k)].q[ O5] = Q[index(i+1, j,   k-1)].q[ O5];
                Q[index(i, j, k)].q[O23] = Q[index(i+1, j,   k+1)].q[O23];
                
                Q[index(i, j, k)].q[ O7] = Q[index(i,   j+1, k-1)].q[ O7];
                Q[index(i, j, k)].q[O25] = Q[index(i,   j+1, k+1)].q[O25];
            }
        }
    }


    for (tNi i=xg1;  i>=1; i--) {
        for (tNi j=yg1;  j>=1; j--) {
            for (tNi k=zg1;  k>=1; k--) {
                //DST   =   SRC

                Q[index(i, j, k)].q[O12] = Q[index(i-1, j,   k  )].q[O12];
                Q[index(i, j, k)].q[O10] = Q[index(i,   j-1, k  )].q[O10];
                Q[index(i, j, k)].q[ O4] = Q[index(i,   j,   k-1)].q[ O4];

                Q[index(i, j, k)].q[ O9] = Q[index(i-1, j-1, k  )].q[ O9];
                Q[index(i, j, k)].q[O15] = Q[index(i-1, j+1, k  )].q[O15];
                
                Q[index(i, j, k)].q[ O3] = Q[index(i-1, j,   k-1)].q[ O3];
                Q[index(i, j, k)].q[O21] = Q[index(i-1, j,   k+1)].q[O21];
                
                Q[index(i, j, k)].q[ O1] = Q[index(i,   j-1, k-1)].q[ O1];
                Q[index(i, j, k)].q[O19] = Q[index(i,   j-1, k+1)].q[O19];
            }
        }
    }
}


