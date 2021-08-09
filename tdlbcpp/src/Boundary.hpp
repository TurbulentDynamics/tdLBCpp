//
//  Boundary.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"





template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, NVecSize, MemoryLayout>::bounceBackBoundary(){

    //Takes the vector from the active cell, reverses it, and places it in the
    //ghost cell (the streaming function can then operate on the ghost cell to
    //bring it back to the active cell

    bounceBackBoundaryRight();
    bounceBackBoundaryLeft();
    bounceBackBoundaryUp();
    bounceBackBoundaryDown();
    bounceBackBoundaryBackward();
    bounceBackBoundaryForward();

    //Needs to be separated into each edge.
    bounceBackEdges();


}





template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, NVecSize, MemoryLayout>::bounceBackEdges(){

    tNi i = 0;
    tNi j = 0;
    tNi k = 0;


    i = 0;
    j = 0;
    for (k=1;  k<=zg1; k++) {
        Q[index(i,j,k)].q[Q07] = Q[dirnQ07(i, j, k)].q[Q08];
    }

    i = xg0;
    j = 0;
    for (k=1;  k<=zg1; k++) {
        Q[index(i,j,k)].q[Q14] = Q[dirnQ14(i, j, k)].q[Q13];
    }

    i = 0;
    j = yg0;
    for (k=1;  k<=zg1; k++) {
        Q[index(i,j,k)].q[Q13] = Q[dirnQ13(i, j, k)].q[Q14];
    }

    i = xg0;
    j = yg0;
    for (k=1;  k<=zg1; k++) {
        Q[index(i,j,k)].q[Q08] = Q[dirnQ08(i, j, k)].q[Q07];
    }



    i = 0;
    k = 0;
    for (j=1;  j<=yg1; j++) {
        Q[index(i,j,k)].q[Q09] = Q[dirnQ09(i, j, k)].q[Q10];
    }


    i = 0;
    k = zg0;
    for (j=1;  j<=yg1; j++) {
        Q[index(i,j,k)].q[Q15] = Q[dirnQ15(i, j, k)].q[Q16];
    }

    i = xg0;
    k = zg0;
    for (j=1;  j<=yg1; j++) {
        Q[index(i,j,k)].q[Q10] = Q[dirnQ10(i, j, k)].q[Q09];
    }

    i = xg0;
    k = 0;
    for (j=1;  j<=yg1; j++) {
        Q[index(i,j,k)].q[Q16] = Q[dirnQ16(i, j, k)].q[Q15];
    }

    j = 0;
    k = 0;
    for (i=1;  i<=xg1; i++) {
        Q[index(i,j,k)].q[Q11] = Q[dirnQ11(i, j, k)].q[Q12];
    }

    j = 0;
    k = zg0;
    for (i=1;  i<=xg1; i++) {
        Q[index(i,j,k)].q[Q17] = Q[dirnQ17(i, j, k)].q[Q18];
    }




    j = yg0;
    k = zg0;
    for (i=1;  i<=xg1; i++) {
        Q[index(i,j,k)].q[Q12] = Q[dirnQ12(i, j, k)].q[Q11];
    }

    j = yg0;
    k = 0;
    for (i=1;  i<=xg1; i++) {
        Q[index(i,j,k)].q[Q18] = Q[dirnQ18(i, j, k)].q[Q17];
    }






}



template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, NVecSize, MemoryLayout>::bounceBackBoundaryRight(){

    //dest = source

    for (tNi j = 1; j<=yg1; j++){
        for (tNi k = 1; k<=zg1; k++){
            tNi i = 0;

            Q[index(i,j,k)].q[Q01] = Q[dirnQ01(i, j, k)].q[Q02];
            Q[index(i,j,k)].q[Q07] = Q[dirnQ07(i, j, k)].q[Q08];
            Q[index(i,j,k)].q[Q13] = Q[dirnQ13(i, j, k)].q[Q14];
            Q[index(i,j,k)].q[Q09] = Q[dirnQ09(i, j, k)].q[Q10];
            Q[index(i,j,k)].q[Q15] = Q[dirnQ15(i, j, k)].q[Q16];

        }
    }
}


template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, NVecSize, MemoryLayout>::bounceBackBoundaryLeft(){

    for (tNi j = 1; j<=yg1; j++){
        for (tNi k = 1; k<=zg1; k++){
            tNi i = xg0;

            Q[index(i,j,k)].q[Q02] = Q[dirnQ02(i, j, k)].q[Q01];
            Q[index(i,j,k)].q[Q08] = Q[dirnQ08(i, j, k)].q[Q07];
            Q[index(i,j,k)].q[Q14] = Q[dirnQ14(i, j, k)].q[Q13];
            Q[index(i,j,k)].q[Q10] = Q[dirnQ10(i, j, k)].q[Q09];
            Q[index(i,j,k)].q[Q16] = Q[dirnQ16(i, j, k)].q[Q15];

        }
    }
}


template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, NVecSize, MemoryLayout>::bounceBackBoundaryUp(){



    //TODO Check xg1 here
    for (tNi i = 1; i<=xg1; i++){
        for (tNi k = 1; k<=zg1; k++){
            tNi j = 0;

            Q[index(i,j,k)].q[Q03] = Q[dirnQ03(i, j, k)].q[Q04];
            Q[index(i,j,k)].q[Q07] = Q[dirnQ07(i, j, k)].q[Q08];
            Q[index(i,j,k)].q[Q14] = Q[dirnQ14(i, j, k)].q[Q13];
            Q[index(i,j,k)].q[Q11] = Q[dirnQ11(i, j, k)].q[Q12];
            Q[index(i,j,k)].q[Q17] = Q[dirnQ17(i, j, k)].q[Q18];

        }
    }
}


template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, NVecSize, MemoryLayout>::bounceBackBoundaryDown(){


    for (tNi i = 1; i<=xg1; i++){
        for (tNi k = 1; k<=zg1; k++){
            tNi j = yg0;

            Q[index(i,j,k)].q[Q04] = Q[dirnQ04(i, j, k)].q[Q03];
            Q[index(i,j,k)].q[Q08] = Q[dirnQ08(i, j, k)].q[Q07];
            Q[index(i,j,k)].q[Q13] = Q[dirnQ13(i, j, k)].q[Q14];
            Q[index(i,j,k)].q[Q12] = Q[dirnQ12(i, j, k)].q[Q11];
            Q[index(i,j,k)].q[Q18] = Q[dirnQ18(i, j, k)].q[Q17];

        }
    }
}


template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, NVecSize, MemoryLayout>::bounceBackBoundaryBackward(){


    for (tNi i = 1; i<=xg1; i++){
        for (tNi j = 1; j<=yg1; j++){
            tNi k = 0;


            Q[index(i,j,k)].q[Q05] = Q[ dirnQ05(i, j, k)].q[Q06];
            Q[index(i,j,k)].q[Q09] = Q[ dirnQ09(i, j, k)].q[Q10];
            Q[index(i,j,k)].q[Q16] = Q[dirnQ16(i, j, k)].q[Q15];
            Q[index(i,j,k)].q[Q11] = Q[dirnQ11(i, j, k)].q[Q12];
            Q[index(i,j,k)].q[Q18] = Q[dirnQ18(i, j, k)].q[Q17];

        }
    }
}


template <typename T, int NVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, NVecSize, MemoryLayout>::bounceBackBoundaryForward(){


    for (tNi i = 1; i<=xg1; i++){
        for (tNi j = 1; j<=yg1; j++){
            tNi k = zg0;

            Q[index(i,j,k)].q[Q06] = Q[dirnQ06(i, j, k)].q[Q05];
            Q[index(i,j,k)].q[Q10] = Q[dirnQ10(i, j, k)].q[Q09];
            Q[index(i,j,k)].q[Q15] = Q[dirnQ15(i, j, k)].q[Q16];
            Q[index(i,j,k)].q[Q12] = Q[dirnQ12(i, j, k)].q[Q11];
            Q[index(i,j,k)].q[Q17] = Q[dirnQ17(i, j, k)].q[Q18];

        }
    }
}
