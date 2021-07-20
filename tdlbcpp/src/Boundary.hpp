//
//  Boundary.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"



//TODO
//J is +ve downwards. Change? Why?






template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnit<T, QVecSize, MemoryLayout>::bounceBackBoundary(){
    
    //Takes the vector from the active cell, reverses it, and places it in the
    //ghost cell (the streaming function can then operate on the ghost cell to
    //bring it back to the active cell
    
    bounceBackBoundaryRight();
    bounceBackBoundaryLeft();
    bounceBackBoundaryUp();
    bounceBackBoundaryDown();
    bounceBackBoundaryBackward();
    bounceBackBoundaryForward();

}






template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnit<T, QVecSize, MemoryLayout>::bounceBackBoundaryRight(){

    //dest = source
    
    for (tNi j = 1; j<=yg1; j++){
        for (tNi k = 1; k<=zg1; k++){
            tNi i = xg1;
            
            //Right
            Q[dirnQ2(i,j,k)].q[Q2] = Q[index(i,j,k)].q[Q1];
            Q[dirnQ8(i,j,k)].q[Q8] = Q[index(i,j,k)].q[Q7];
            Q[dirnQ10(i,j,k)].q[Q10] = Q[index(i,j,k)].q[Q9];
            Q[dirnQ14(i,j,k)].q[Q14] = Q[index(i,j,k)].q[Q13];
            Q[dirnQ16(i,j,k)].q[Q16] = Q[index(i,j,k)].q[Q15];
            Q[dirnQ20(i,j,k)].q[Q20] = Q[index(i,j,k)].q[Q19];
            Q[dirnQ22(i,j,k)].q[Q22] = Q[index(i,j,k)].q[Q21];
            Q[dirnQ24(i,j,k)].q[Q24] = Q[index(i,j,k)].q[Q23];
            Q[dirnQ25(i,j,k)].q[Q25] = Q[index(i,j,k)].q[Q26];
        }
    }
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnit<T, QVecSize, MemoryLayout>::bounceBackBoundaryLeft(){
    
    for (tNi j = 1; j<=yg1; j++){
        for (tNi k = 1; k<=zg1; k++){
            tNi i = 1;
            
            //Left
            Q[dirnQ1(i,j,k)].q[Q1] = Q[index(i,j,k)].q[Q2];
            Q[dirnQ7(i,j,k)].q[Q7] = Q[index(i,j,k)].q[Q8];
            Q[dirnQ9(i,j,k)].q[Q9] = Q[index(i,j,k)].q[Q10];
            Q[dirnQ13(i,j,k)].q[Q13] = Q[index(i,j,k)].q[Q14];
            Q[dirnQ15(i,j,k)].q[Q15] = Q[index(i,j,k)].q[Q16];
            Q[dirnQ19(i,j,k)].q[Q19] = Q[index(i,j,k)].q[Q20];
            Q[dirnQ21(i,j,k)].q[Q21] = Q[index(i,j,k)].q[Q22];
            Q[dirnQ23(i,j,k)].q[Q23] = Q[index(i,j,k)].q[Q24];
            Q[dirnQ26(i,j,k)].q[Q26] = Q[index(i,j,k)].q[Q25];
        }
    }
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnit<T, QVecSize, MemoryLayout>::bounceBackBoundaryUp(){
    
    
    
    //TODO Check xg1 here
    for (tNi i = 1; i<=xg1; i++){
        for (tNi k = 1; k<=zg1; k++){
            tNi j = 1;
            
            //Up
            Q[dirnQ4(i,j,k)].q[Q4] = Q[index(i,j,k)].q[Q3];
            Q[dirnQ8(i,j,k)].q[Q8] = Q[index(i,j,k)].q[Q7];
            Q[dirnQ12(i,j,k)].q[Q12] = Q[index(i,j,k)].q[Q11];
            Q[dirnQ13(i,j,k)].q[Q13] = Q[index(i,j,k)].q[Q14];
            Q[dirnQ18(i,j,k)].q[Q18] = Q[index(i,j,k)].q[Q17];
            Q[dirnQ20(i,j,k)].q[Q20] = Q[index(i,j,k)].q[Q19];
            Q[dirnQ22(i,j,k)].q[Q22] = Q[index(i,j,k)].q[Q21];
            Q[dirnQ23(i,j,k)].q[Q23] = Q[index(i,j,k)].q[Q24];
            Q[dirnQ26(i,j,k)].q[Q26] = Q[index(i,j,k)].q[Q25];
        }
    }
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnit<T, QVecSize, MemoryLayout>::bounceBackBoundaryDown(){
    
    
    for (tNi i = 1; i<=xg1; i++){
        for (tNi k = 1; k<=zg1; k++){
            tNi j = 1;
            
            //Down
            Q[dirnQ3(i,j,k)].q[Q3] = Q[index(i,j,k)].q[Q4];
            Q[dirnQ7(i,j,k)].q[Q7] = Q[index(i,j,k)].q[Q8];
            Q[dirnQ11(i,j,k)].q[Q11] = Q[index(i,j,k)].q[Q12];
            Q[dirnQ14(i,j,k)].q[Q14] = Q[index(i,j,k)].q[Q13];
            Q[dirnQ17(i,j,k)].q[Q17] = Q[index(i,j,k)].q[Q18];
            Q[dirnQ19(i,j,k)].q[Q19] = Q[index(i,j,k)].q[Q20];
            Q[dirnQ21(i,j,k)].q[Q21] = Q[index(i,j,k)].q[Q22];
            Q[dirnQ24(i,j,k)].q[Q24] = Q[index(i,j,k)].q[Q23];
            Q[dirnQ25(i,j,k)].q[Q25] = Q[index(i,j,k)].q[Q26];
        }
    }
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnit<T, QVecSize, MemoryLayout>::bounceBackBoundaryBackward(){
    
    
    for (tNi i = 1; i<=xg1; i++){
        for (tNi j = 1; j<=yg1; j++){
            tNi k = 1;
            
            //Forwards
            Q[dirnQ6(i,j,k)].q[Q6] = Q[index(i,j,k)].q[Q5];
            Q[dirnQ10(i,j,k)].q[Q10] = Q[index(i,j,k)].q[Q9];
            Q[dirnQ12(i,j,k)].q[Q12] = Q[index(i,j,k)].q[Q11];
            Q[dirnQ15(i,j,k)].q[Q15] = Q[index(i,j,k)].q[Q16];
            Q[dirnQ17(i,j,k)].q[Q17] = Q[index(i,j,k)].q[Q18];
            Q[dirnQ20(i,j,k)].q[Q20] = Q[index(i,j,k)].q[Q19];
            Q[dirnQ21(i,j,k)].q[Q21] = Q[index(i,j,k)].q[Q22];
            Q[dirnQ24(i,j,k)].q[Q24] = Q[index(i,j,k)].q[Q23];
            Q[dirnQ26(i,j,k)].q[Q26] = Q[index(i,j,k)].q[Q25];
        }
    }
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnit<T, QVecSize, MemoryLayout>::bounceBackBoundaryForward(){
    
    
    for (tNi i = 1; i<=xg1; i++){
        for (tNi j = 1; j<=yg1; j++){
            tNi k = zg1;
            
            //Backwards
            Q[dirnQ5(i,j,k)].q[Q5] = Q[index(i,j,k)].q[Q6];
            Q[dirnQ9(i,j,k)].q[Q9] = Q[index(i,j,k)].q[Q10];
            Q[dirnQ11(i,j,k)].q[Q11] = Q[index(i,j,k)].q[Q12];
            Q[dirnQ16(i,j,k)].q[Q16] = Q[index(i,j,k)].q[Q15];
            Q[dirnQ18(i,j,k)].q[Q18] = Q[index(i,j,k)].q[Q17];
            Q[dirnQ19(i,j,k)].q[Q19] = Q[index(i,j,k)].q[Q20];
            Q[dirnQ22(i,j,k)].q[Q22] = Q[index(i,j,k)].q[Q21];
            Q[dirnQ23(i,j,k)].q[Q23] = Q[index(i,j,k)].q[Q24];
            Q[dirnQ25(i,j,k)].q[Q25] = Q[index(i,j,k)].q[Q26];
        }
    }
}
