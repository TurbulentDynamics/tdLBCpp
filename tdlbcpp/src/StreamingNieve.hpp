//
//  StreamingNieve.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"







template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
void ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Simple>::streaming(){
    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){
                //DST    =  SRC
                // moving towards Q5, then Q3, then Q1
                // when moving we should not overwrite old values
                Q[index(i, j, k)].q[ Q2] = Q[ dirnQ1(i,j,k)].q[ Q2];
                Q[index(i, j, k)].q[ Q4] = Q[ dirnQ3(i,j,k)].q[ Q4];
                Q[index(i, j, k)].q[ Q6] = Q[ dirnQ5(i,j,k)].q[ Q6];

                Q[index(i, j, k)].q[Q14] = Q[dirnQ13(i,j,k)].q[Q14];
                Q[index(i, j, k)].q[ Q8] = Q[ dirnQ7(i,j,k)].q[ Q8];
                
                Q[index(i, j, k)].q[Q16] = Q[dirnQ15(i,j,k)].q[Q16];
                Q[index(i, j, k)].q[Q10] = Q[ dirnQ9(i,j,k)].q[Q10];
                
                Q[index(i, j, k)].q[Q18] = Q[dirnQ17(i,j,k)].q[Q18];
                Q[index(i, j, k)].q[Q12] = Q[dirnQ11(i,j,k)].q[Q12];
            }
        }
    }


    for (tNi i=xg1;  i>=1; i--) {
        for (tNi j=yg1;  j>=1; j--) {
            for (tNi k=zg1;  k>=1; k--) {
                
                //DST   =   SRC
                Q[index(i, j, k)].q[ Q1] = Q[ dirnQ2(i,j,k)].q[ Q1];
                Q[index(i, j, k)].q[ Q3] = Q[ dirnQ4(i,j,k)].q[ Q3];
                Q[index(i, j, k)].q[ Q5] = Q[ dirnQ6(i,j,k)].q[ Q5];

                Q[index(i, j, k)].q[ Q7] = Q[ dirnQ8(i,j,k)].q[ Q7];
                Q[index(i, j, k)].q[Q13] = Q[dirnQ14(i,j,k)].q[Q13];
                
                Q[index(i, j, k)].q[ Q9] = Q[dirnQ10(i,j,k)].q[ Q9];
                Q[index(i, j, k)].q[Q15] = Q[dirnQ16(i,j,k)].q[Q15];
                
                Q[index(i, j, k)].q[Q11] = Q[dirnQ12(i,j,k)].q[Q11];
                Q[index(i, j, k)].q[Q17] = Q[dirnQ18(i,j,k)].q[Q17];
            }
        }
    }
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
void ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Simple>::streaming2(){
    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){

                //DST    =  SRC
                Q[ dirnQ2(i,j,k)].q[ Q2] = Q[index(i, j, k)].q[ Q2];
                Q[ dirnQ4(i,j,k)].q[ Q4] = Q[index(i, j, k)].q[ Q4];
                Q[ dirnQ6(i,j,k)].q[ Q6] = Q[index(i, j, k)].q[ Q6];


                Q[dirnQ14(i,j,k)].q[Q14] = Q[index(i, j, k)].q[Q14];
                Q[ dirnQ8(i,j,k)].q[ Q8] = Q[index(i, j, k)].q[ Q8];

                Q[dirnQ16(i,j,k)].q[Q16] = Q[index(i, j, k)].q[Q16];
                Q[dirnQ10(i,j,k)].q[Q10] = Q[index(i, j, k)].q[Q10];

                Q[dirnQ18(i,j,k)].q[Q18] = Q[index(i, j, k)].q[Q18];
                Q[dirnQ12(i,j,k)].q[Q12] = Q[index(i, j, k)].q[Q12];

            }
        }
    }


    for (tNi i=xg1;  i>=1; i--) {
        for (tNi j=yg1;  j>=1; j--) {
            for (tNi k=zg1;  k>=1; k--) {

                //DST   =   SRC
                Q[ dirnQ1(i,j,k)].q[ Q1] = Q[index(i, j, k)].q[ Q1];
                Q[ dirnQ3(i,j,k)].q[ Q3] = Q[index(i, j, k)].q[ Q3];
                Q[ dirnQ5(i,j,k)].q[ Q5] = Q[index(i, j, k)].q[ Q5];


                Q[ dirnQ7(i,j,k)].q[ Q7] = Q[index(i, j, k)].q[ Q7];
                Q[dirnQ13(i,j,k)].q[Q13] = Q[index(i, j, k)].q[Q13];

                Q[ dirnQ9(i,j,k)].q[ Q9] = Q[index(i, j, k)].q[ Q9];
                Q[dirnQ15(i,j,k)].q[Q15] = Q[index(i, j, k)].q[Q15];

                Q[dirnQ11(i,j,k)].q[Q11] = Q[index(i, j, k)].q[Q11];
                Q[dirnQ17(i,j,k)].q[Q17] = Q[index(i, j, k)].q[Q17];

            }
        }
    }
}




template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
void ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Simple>::streamingDEBUG(){

    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){
                //DST    =  SRC
                Q[index(i, j, k)].q[ Q3] = Q[index(i+1, j,   k  )].q[ Q3];
                Q[index(i, j, k)].q[ Q4] = Q[index(i,   j+1, k  )].q[ Q4];
                Q[index(i, j, k)].q[ Q6] = Q[index(i,   j,   k+1)].q[ Q6];

                Q[index(i, j, k)].q[ Q8] = Q[index(i+1, j-1, k  )].q[ Q8];
                Q[index(i, j, k)].q[ Q9] = Q[index(i+1, j+1, k  )].q[ Q9];

                Q[index(i, j, k)].q[Q12] = Q[index(i+1, j,   k-1)].q[Q12];
                Q[index(i, j, k)].q[Q13] = Q[index(i+1, j,   k+1)].q[Q13];

                Q[index(i, j, k)].q[Q16] = Q[index(i,   j+1, k-1)].q[Q16];
                Q[index(i, j, k)].q[Q17] = Q[index(i,   j+1, k+1)].q[Q17];
            }
        }
    }


    for (tNi i=xg1;  i>=1; i--) {
        for (tNi j=yg1;  j>=1; j--) {
            for (tNi k=zg1;  k>=1; k--) {
                //DST   =   SRC

                Q[index(i, j, k)].q[ Q1] = Q[index(i-1, j,   k  )].q[ Q1];
                Q[index(i, j, k)].q[ Q2] = Q[index(i,   j-1, k  )].q[ Q2];
                Q[index(i, j, k)].q[ Q5] = Q[index(i,   j,   k-1)].q[ Q5];

                Q[index(i, j, k)].q[ Q7] = Q[index(i-1, j-1, k  )].q[ Q7];
                Q[index(i, j, k)].q[Q10] = Q[index(i-1, j+1, k  )].q[Q10];

                Q[index(i, j, k)].q[Q11] = Q[index(i-1, j,   k-1)].q[Q11];
                Q[index(i, j, k)].q[Q14] = Q[index(i-1, j,   k+1)].q[Q14];

                Q[index(i, j, k)].q[Q15] = Q[index(i,   j-1, k-1)].q[Q15];
                Q[index(i, j, k)].q[Q18] = Q[index(i,   j-1, k+1)].q[Q18];
            }
        }
    }
}

