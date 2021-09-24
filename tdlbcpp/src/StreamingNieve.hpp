//
//  StreamingNieve.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"







template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
void ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Simple>::streamingPull(){


    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){

                //DST    =  SRC
                Q[index(i, j, k)].q[Q02] = Q[dirnQ01(i,j,k)].q[Q02];
                Q[index(i, j, k)].q[Q04] = Q[dirnQ03(i,j,k)].q[Q04];
                Q[index(i, j, k)].q[Q06] = Q[dirnQ05(i,j,k)].q[Q06];

                Q[index(i, j, k)].q[Q08] = Q[dirnQ07(i,j,k)].q[Q08];
                Q[index(i, j, k)].q[Q10] = Q[dirnQ09(i,j,k)].q[Q10];

                Q[index(i, j, k)].q[Q12] = Q[dirnQ11(i,j,k)].q[Q12];
                Q[index(i, j, k)].q[Q14] = Q[dirnQ13(i,j,k)].q[Q14];

                Q[index(i, j, k)].q[Q16] = Q[dirnQ15(i,j,k)].q[Q16];
                Q[index(i, j, k)].q[Q18] = Q[dirnQ17(i,j,k)].q[Q18];

            }
        }
    }


    for (tNi i=xg1;  i>=1; i--) {
        for (tNi j=yg1;  j>=1; j--) {
            for (tNi k=zg1;  k>=1; k--) {

                //DST   =   SRC
                Q[index(i, j, k)].q[Q01] = Q[dirnQ02(i,j,k)].q[Q01];
                Q[index(i, j, k)].q[Q03] = Q[dirnQ04(i,j,k)].q[Q03];
                Q[index(i, j, k)].q[Q05] = Q[dirnQ06(i,j,k)].q[Q05];

                Q[index(i, j, k)].q[Q07] = Q[dirnQ08(i,j,k)].q[Q07];
                Q[index(i, j, k)].q[Q09] = Q[dirnQ10(i,j,k)].q[Q09];

                Q[index(i, j, k)].q[Q11] = Q[dirnQ12(i,j,k)].q[Q11];
                Q[index(i, j, k)].q[Q13] = Q[dirnQ14(i,j,k)].q[Q13];

                Q[index(i, j, k)].q[Q15] = Q[dirnQ16(i,j,k)].q[Q15];
                Q[index(i, j, k)].q[Q17] = Q[dirnQ18(i,j,k)].q[Q17];

            }
        }
    }
}









template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
void ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Simple>::streamingPush(){


    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){

                //DST    =  SRC
                Q[dirnQ02(i,j,k)].q[Q02] = Q[index(i, j, k)].q[Q02];
                Q[dirnQ04(i,j,k)].q[Q04] = Q[index(i, j, k)].q[Q04];
                Q[dirnQ06(i,j,k)].q[Q06] = Q[index(i, j, k)].q[Q06];


                Q[dirnQ08(i,j,k)].q[Q08] = Q[index(i, j, k)].q[Q08];
                Q[dirnQ10(i,j,k)].q[Q10] = Q[index(i, j, k)].q[Q10];

                Q[dirnQ12(i,j,k)].q[Q12] = Q[index(i, j, k)].q[Q12];
                Q[dirnQ14(i,j,k)].q[Q14] = Q[index(i, j, k)].q[Q14];

                Q[dirnQ16(i,j,k)].q[Q16] = Q[index(i, j, k)].q[Q16];
                Q[dirnQ18(i,j,k)].q[Q18] = Q[index(i, j, k)].q[Q18];

            }
        }
    }


    for (tNi i=xg1;  i>=1; i--) {
        for (tNi j=yg1;  j>=1; j--) {
            for (tNi k=zg1;  k>=1; k--) {

                //DST   =   SRC
                Q[dirnQ01(i,j,k)].q[Q01] = Q[index(i, j, k)].q[Q01];
                Q[dirnQ03(i,j,k)].q[Q03] = Q[index(i, j, k)].q[Q03];
                Q[dirnQ05(i,j,k)].q[Q05] = Q[index(i, j, k)].q[Q05];


                Q[dirnQ07(i,j,k)].q[Q07] = Q[index(i, j, k)].q[Q07];
                Q[dirnQ09(i,j,k)].q[Q09] = Q[index(i, j, k)].q[Q09];

                Q[dirnQ11(i,j,k)].q[Q11] = Q[index(i, j, k)].q[Q11];
                Q[dirnQ13(i,j,k)].q[Q13] = Q[index(i, j, k)].q[Q13];

                Q[dirnQ15(i,j,k)].q[Q15] = Q[index(i, j, k)].q[Q15];
                Q[dirnQ17(i,j,k)].q[Q17] = Q[index(i, j, k)].q[Q17];

            }
        }
    }
}




