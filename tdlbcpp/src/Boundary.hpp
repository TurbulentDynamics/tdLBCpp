//
//  Boundary.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"



//TODO
//J is +ve downwards.






template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::bounceBackBoundary(){

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





template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::bounceBackEdges(){

    tNi i = 0;
    tNi j = 0;
    tNi k = 0;
    for (k=1;  k<=zg1; k++) {
        Q[index(i,j,k)].q[ Q7] = Q[index( i+1, j+1, k  )].q[ Q9];
    }


    i = xg0;
    j = 0;
    for (k=1;  k<=zg1; k++) {
        Q[index(i,j,k)].q[ Q8] = Q[index( i-1, j+1, k  )].q[ Q10];
    }

    i = 0;
    j = yg0;
    for (k=1;  k<=zg1; k++) {
        Q[index(i,j,k)].q[Q10] = Q[index( i+1, j-1, k  )].q[ Q8];
    }

    i = xg0;
    j = yg0;
    for (k=1;  k<=zg1; k++) {
        Q[index(i,j,k)].q[ Q9] = Q[index( i-1, j-1, k  )].q[ Q7];
    }



    i = 0;
    k = 0;
    for (j=1;  j<=yg1; j++) {
        Q[index(i,j,k)].q[Q11] = Q[index( i+1, j  , k+1)].q[ Q13];
    }


    i = 0;
    k = zg0;
    for (j=1;  j<=yg1; j++) {
        Q[index(i,j,k)].q[Q14] = Q[index( i+1, j  , k-1)].q[ Q12];
    }

    i = xg0;
    k = zg0;
    for (j=1;  j<=yg1; j++) {
        Q[index(i,j,k)].q[Q13] = Q[index( i-1, j  , k-1)].q[ Q11];
    }

    i = xg0;
    k = 0;
    for (j=1;  j<=yg1; j++) {
        Q[index(i,j,k)].q[Q12] = Q[index( i-1, j  , k+1)].q[ Q14];
    }

    j = 0;
    k = 0;
    for (i=1;  i<=xg1; i++) {
        Q[index(i,j,k)].q[Q15] = Q[index( i  , j+1, k+1)].q[ Q17];
    }

    j = 0;
    k = zg0;
    for (i=1;  i<=xg1; i++) {
        Q[index(i,j,k)].q[Q18] = Q[index( i  , j+1, k-1)].q[ Q16];
    }




    j = yg0;
    k = zg0;
    for (i=1;  i<=xg1; i++) {
        Q[index(i,j,k)].q[Q17] = Q[index( i  , j-1, k-1)].q[ Q15];
    }

    j = yg0;
    k = 0;
    for (i=1;  i<=xg1; i++) {
        Q[index(i,j,k)].q[Q16] = Q[index( i  , j-1, k+1)].q[ Q18];
    }






}



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::bounceBackBoundaryRight(){

    //dest = source

    for (tNi j = 1; j<=yg1; j++){
        for (tNi k = 1; k<=zg1; k++){
            tNi i = 0;

            Q[index(i,j,k)].q[ Q1] = Q[index( i+1, j  , k  )].q[  Q3];
            Q[index(i,j,k)].q[ Q7] = Q[index( i+1, j+1, k  )].q[  Q9];
            Q[index(i,j,k)].q[Q10] = Q[index( i+1, j-1, k  )].q[  Q8];
            Q[index(i,j,k)].q[Q11] = Q[index( i+1, j  , k+1)].q[ Q13];
            Q[index(i,j,k)].q[Q14] = Q[index( i+1, j  , k-1)].q[ Q12];

        }
    }
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::bounceBackBoundaryLeft(){

    for (tNi j = 1; j<=yg1; j++){
        for (tNi k = 1; k<=zg1; k++){
            tNi i = xg0;

            Q[index(i,j,k)].q[ Q3] = Q[index( i-1, j  , k  )].q[ Q1];
            Q[index(i,j,k)].q[ Q9] = Q[index( i-1, j-1, k  )].q[ Q7];
            Q[index(i,j,k)].q[ Q8] = Q[index( i-1, j+1, k  )].q[ Q10];
            Q[index(i,j,k)].q[Q13] = Q[index( i-1, j  , k-1)].q[ Q11];
            Q[index(i,j,k)].q[Q12] = Q[index( i-1, j  , k+1)].q[ Q14];

        }
    }
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::bounceBackBoundaryUp(){



    //TODO Check xg1 here
    for (tNi i = 1; i<=xg1; i++){
        for (tNi k = 1; k<=zg1; k++){
            tNi j = 0;

            Q[index(i,j,k)].q[ Q2] = Q[index( i  , j+1, k  )].q[ Q4];
            Q[index(i,j,k)].q[ Q7] = Q[index( i+1, j+1, k  )].q[ Q9];
            Q[index(i,j,k)].q[ Q8] = Q[index( i-1, j+1, k  )].q[ Q10];
            Q[index(i,j,k)].q[Q15] = Q[index( i  , j+1, k+1)].q[ Q17];
            Q[index(i,j,k)].q[Q18] = Q[index( i  , j+1, k-1)].q[ Q16];

        }
    }
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::bounceBackBoundaryDown(){


    for (tNi i = 1; i<=xg1; i++){
        for (tNi k = 1; k<=zg1; k++){
            tNi j = yg0;

            Q[index(i,j,k)].q[ Q4] = Q[index( i  , j-1, k  )].q[ Q2];
            Q[index(i,j,k)].q[ Q9] = Q[index( i-1, j-1, k  )].q[ Q7];
            Q[index(i,j,k)].q[Q10] = Q[index( i+1, j-1, k  )].q[ Q8];
            Q[index(i,j,k)].q[Q17] = Q[index( i  , j-1, k-1)].q[ Q15];
            Q[index(i,j,k)].q[Q16] = Q[index( i  , j-1, k+1)].q[ Q18];

        }
    }
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::bounceBackBoundaryBackward(){


    for (tNi i = 1; i<=xg1; i++){
        for (tNi j = 1; j<=yg1; j++){
            tNi k = 0;


            Q[index(i,j,k)].q[ Q5] = Q[index( i  , j  , k+1)].q[ Q6];
            Q[index(i,j,k)].q[Q11] = Q[index( i+1, j  , k+1)].q[ Q13];
            Q[index(i,j,k)].q[Q12] = Q[index( i-1, j  , k+1)].q[ Q14];
            Q[index(i,j,k)].q[Q15] = Q[index( i  , j+1, k+1)].q[ Q17];
            Q[index(i,j,k)].q[Q16] = Q[index( i  , j-1, k+1)].q[ Q18];

        }
    }
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::bounceBackBoundaryForward(){


    for (tNi i = 1; i<=xg1; i++){
        for (tNi j = 1; j<=yg1; j++){
            tNi k = zg0;

            Q[index(i,j,k)].q[ Q6] = Q[index( i  , j  , k-1)].q[ Q5];
            Q[index(i,j,k)].q[Q13] = Q[index( i-1, j  , k-1)].q[ Q11];
            Q[index(i,j,k)].q[Q14] = Q[index( i+1, j  , k-1)].q[ Q12];
            Q[index(i,j,k)].q[Q17] = Q[index( i  , j-1, k-1)].q[ Q15];
            Q[index(i,j,k)].q[Q18] = Q[index( i  , j+1, k-1)].q[ Q16];

        }
    }
}
