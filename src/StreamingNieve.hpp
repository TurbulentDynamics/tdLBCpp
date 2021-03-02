//
//  StreamingNieve.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"







template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::streaming_simple(){
    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){
                //DST    =  SRC
                Q[index(i, j, k)].q[ Q1] = Q[dirnQ1(i,j,k)].q[ Q1];
                Q[index(i, j, k)].q[ Q4] = Q[dirnQ4(i,j,k)].q[ Q4];
                Q[index(i, j, k)].q[ Q5] = Q[dirnQ5(i,j,k)].q[ Q5];

                Q[index(i, j, k)].q[Q13] = Q[dirnQ13(i,j,k)].q[Q13];
                Q[index(i, j, k)].q[ Q7] = Q[dirnQ7(i,j,k)].q[ Q7];
                
                Q[index(i, j, k)].q[Q15] = Q[dirnQ15(i,j,k)].q[Q15];
                Q[index(i, j, k)].q[ Q9] = Q[dirnQ9(i,j,k)].q[ Q9];
                
                Q[index(i, j, k)].q[Q17] = Q[dirnQ17(i,j,k)].q[Q17];
                Q[index(i, j, k)].q[Q11] = Q[dirnQ11(i,j,k)].q[Q11];
            }
        }
    }


    for (tNi i=xg1;  i>=1; i--) {
        for (tNi j=yg1;  j>=1; j--) {
            for (tNi k=zg1;  k>=1; k--) {
                
                //DST   =   SRC
                Q[index(i, j, k)].q[ Q2] = Q[dirnQ2(i,j,k)].q[ Q2];
                Q[index(i, j, k)].q[ Q3] = Q[dirnQ3(i,j,k)].q[ Q3];
                Q[index(i, j, k)].q[ Q6] = Q[dirnQ6(i,j,k)].q[ Q6];

                Q[index(i, j, k)].q[ Q8] = Q[dirnQ8(i,j,k)].q[ Q8];
                Q[index(i, j, k)].q[Q14] = Q[dirnQ14(i,j,k)].q[Q14];
                
                Q[index(i, j, k)].q[Q10] = Q[dirnQ10(i,j,k)].q[Q10];
                Q[index(i, j, k)].q[Q16] = Q[dirnQ16(i,j,k)].q[Q16];
                
                Q[index(i, j, k)].q[Q12] = Q[dirnQ12(i,j,k)].q[Q12];
                Q[index(i, j, k)].q[Q18] = Q[dirnQ18(i,j,k)].q[Q18];
            }
        }
    }
}





//template <typename T, int QVecSize>
//void ComputeUnit<T, QVecSize>::streaming_simple(){
//    for (tNi i=1; i<=xg1; i++){
//        for (tNi j=1; j<=yg1; j++){
//            for (tNi k=1; k<=zg1; k++){
//                //DST    =  SRC
//                Q[index(i, j, k)].q[O14] = Q[index(i+1, j,   k  )].q[O14];
//                Q[index(i, j, k)].q[O16] = Q[index(i,   j+1, k  )].q[O16];
//                Q[index(i, j, k)].q[O22] = Q[index(i,   j,   k+1)].q[O22];
//
//                Q[index(i, j, k)].q[O11] = Q[index(i+1, j-1, k  )].q[O11];
//                Q[index(i, j, k)].q[O17] = Q[index(i+1, j+1, k  )].q[O17];
//
//                Q[index(i, j, k)].q[ O5] = Q[index(i+1, j,   k-1)].q[ O5];
//                Q[index(i, j, k)].q[O23] = Q[index(i+1, j,   k+1)].q[O23];
//
//                Q[index(i, j, k)].q[ O7] = Q[index(i,   j+1, k-1)].q[ O7];
//                Q[index(i, j, k)].q[O25] = Q[index(i,   j+1, k+1)].q[O25];
//            }
//        }
//    }
//
//
//    for (tNi i=xg1;  i>=1; i--) {
//        for (tNi j=yg1;  j>=1; j--) {
//            for (tNi k=zg1;  k>=1; k--) {
//                //DST   =   SRC
//
//                Q[index(i, j, k)].q[O12] = Q[index(i-1, j,   k  )].q[O12];
//                Q[index(i, j, k)].q[O10] = Q[index(i,   j-1, k  )].q[O10];
//                Q[index(i, j, k)].q[ O4] = Q[index(i,   j,   k-1)].q[ O4];
//
//                Q[index(i, j, k)].q[ O9] = Q[index(i-1, j-1, k  )].q[ O9];
//                Q[index(i, j, k)].q[O15] = Q[index(i-1, j+1, k  )].q[O15];
//
//                Q[index(i, j, k)].q[ O3] = Q[index(i-1, j,   k-1)].q[ O3];
//                Q[index(i, j, k)].q[O21] = Q[index(i-1, j,   k+1)].q[O21];
//
//                Q[index(i, j, k)].q[ O1] = Q[index(i,   j-1, k-1)].q[ O1];
//                Q[index(i, j, k)].q[O19] = Q[index(i,   j-1, k+1)].q[O19];
//            }
//        }
//    }
//}


