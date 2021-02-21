//
//  ComputeGroup.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"







template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::collision_EgglesSomers_LES(){

//    alf2[ 5] = (2.0 * (q[2] + 0.5 * f.x) * u.x - q[ 5]*c)*b * Nue.getNue(i,j,k);

}


template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::collision_EgglesSomers(){
  

    //kinematic viscosity.
    T b = 1.0 / (1.0 + 6 * flow.nu);
    T c = 1.0 - 6 * flow.nu;
        
    
    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){


                Force<T> f = F[index(i,j,k)];

				QVec<T, QVecSize> q = Q[index(i, j, k)];

				Velocity<T> u = q.velocity(f);

                QVec<T, QVecSize> alf2;


                //0th order term
                alf2[ Q1] = q[Q1];


                //1st order term
                alf2[ Q2] = q[Q2] + f.x;
                alf2[ Q3] = q[Q3] + f.y;
                alf2[ Q4] = q[Q4] + f.z;

                //2nd order terms
                alf2[ Q5] = (2.0 * (q[Q2] + 0.5 * f.x) * u.x - q[ Q5]*c)*b;
                alf2[ Q6] = (2.0 * (q[Q2] + 0.5 * f.x) * u.y - q[ Q6]*c)*b;
                alf2[ Q7] = (2.0 * (q[Q3] + 0.5 * f.y) * u.y - q[ Q7]*c)*b;
                alf2[ Q8] = (2.0 * (q[Q2] + 0.5 * f.x) * u.z - q[ Q8]*c)*b;
                alf2[ Q9] = (2.0 * (q[Q3] + 0.5 * f.y) * u.z - q[ Q9]*c)*b;
                alf2[Q10] = (2.0 * (q[Q4] + 0.5 * f.z) * u.z - q[Q10]*c)*b;

                //3rd order terms
                alf2[Q11] =  -flow.g3 * q[Q11];
                alf2[Q12] =  -flow.g3 * q[Q12];
                alf2[Q13] =  -flow.g3 * q[Q13];
                alf2[Q14] =  -flow.g3 * q[Q14];
                alf2[Q15] =  -flow.g3 * q[Q15];
                alf2[Q16] =  -flow.g3 * q[Q16];

                //4th order terms
                alf2[Q17] = 0.0;
                alf2[Q18] = 0.0;


                // Start of invMoments, which is responsible for determining
                // the N-field (x) from alpha+ (alf2). It does this by using eq.
                // 12 in the article by Eggels and Somers (1995), which means
                // it's using the "filter matrix E" (not really present in the
                // code as matrix, but it's where the coefficients come from).

                for (int l=0;  l<QVecSize; l++) {
                    alf2[l] /= 24.0;
                }


                q[Q1]  = 2 * (alf2[Q1] + 2*alf2[Q2] + 1.5*alf2[Q5] - 1.5*alf2[Q7] - 1.5*alf2[Q10] - alf2[Q11] - alf2[Q13] + alf2[Q17] + alf2[Q18]);

                q[Q2]  = 2 * (alf2[Q1] + 2*alf2[Q3] - 1.5*alf2[Q5] + 1.5*alf2[Q7] - 1.5*alf2[Q10] - alf2[Q12] - alf2[Q14] + alf2[Q17] - alf2[Q18]);

                q[Q3]  = 2 * (alf2[Q1] - 2*alf2[Q2] + 1.5*alf2[Q5] - 1.5*alf2[Q7] - 1.5*alf2[Q10] + alf2[Q11] + alf2[Q13] + alf2[Q17] + alf2[Q18]);

                q[Q4]  = 2 * (alf2[Q1] - 2*alf2[Q3] - 1.5*alf2[Q5] + 1.5*alf2[Q7] - 1.5*alf2[Q10] + alf2[Q12] + alf2[Q14] + alf2[Q17] - alf2[Q18]);

                q[Q5]  = 2 * (alf2[Q1] + 2*alf2[Q4] - 1.5*alf2[Q5] - 1.5*alf2[Q7] + 1.5*alf2[Q10] - 2*alf2[Q15] - 2*alf2[Q17]);

                q[Q6]  = 2 * (alf2[Q1] - 2*alf2[Q4] - 1.5*alf2[Q5] - 1.5*alf2[Q7] + 1.5*alf2[Q10] + 2*alf2[Q15] - 2*alf2[Q17]);

                q[Q7]  = alf2[Q1] + 2*alf2[Q2] + 2*alf2[Q3] + 1.5*alf2[Q5] + 6*alf2[Q6] + 1.5*alf2[Q7] - 1.5*alf2[Q10] + 2*alf2[Q11] + 2*alf2[Q12] - 2*alf2[Q17];

                q[Q8]  = alf2[Q1] - 2*alf2[Q2] + 2*alf2[Q3] + 1.5*alf2[Q5] - 6*alf2[Q6] + 1.5*alf2[Q7] - 1.5*alf2[Q10] - 2*alf2[Q11] + 2*alf2[Q12] - 2*alf2[Q17];

                q[Q9]  = alf2[Q1] - 2*alf2[Q2] - 2*alf2[Q3] + 1.5*alf2[Q5] + 6*alf2[Q6] + 1.5*alf2[Q7] - 1.5*alf2[Q10] - 2*alf2[Q11] - 2*alf2[Q12] - 2*alf2[Q17];

                q[Q10] = alf2[Q1] + 2*alf2[Q2] - 2*alf2[Q3] + 1.5*alf2[Q5] - 6*alf2[Q6] + 1.5*alf2[Q7] - 1.5*alf2[Q10] + 2*alf2[Q11] - 2*alf2[Q12] - 2*alf2[Q17];

                q[Q11] = alf2[Q1] + 2*alf2[Q2] + 2*alf2[Q4] + 1.5*alf2[Q5] - 1.5*alf2[Q7] + 6*alf2[Q8] + 1.5*alf2[Q10] - alf2[Q11] + alf2[Q13] + alf2[Q15] - alf2[Q16] + alf2[Q17] - alf2[Q18];

                q[Q12] = alf2[Q1] - 2*alf2[Q2] + 2*alf2[Q4] + 1.5*alf2[Q5] - 1.5*alf2[Q7] - 6*alf2[Q8] + 1.5*alf2[Q10] + alf2[Q11] - alf2[Q13] + alf2[Q15] - alf2[Q16] + alf2[Q17] - alf2[Q18];

                q[Q13] = alf2[Q1] - 2*alf2[Q2] - 2*alf2[Q4] + 1.5*alf2[Q5] - 1.5*alf2[Q7] + 6*alf2[Q8] + 1.5*alf2[Q10] + alf2[Q11] - alf2[Q13] - alf2[Q15] + alf2[Q16] + alf2[Q17] - alf2[Q18];

                q[Q14] = alf2[Q1] + 2*alf2[Q2] - 2*alf2[Q4] + 1.5*alf2[Q5] - 1.5*alf2[Q7] - 6*alf2[Q8] + 1.5*alf2[Q10] - alf2[Q11] + alf2[Q13] - alf2[Q15] + alf2[Q16] + alf2[Q17] - alf2[Q18];

                q[Q15] = alf2[Q1] + 2*alf2[Q3] + 2*alf2[Q4] - 1.5*alf2[Q5] + 1.5*alf2[Q7] + 6*alf2[Q9] + 1.5*alf2[Q10] - alf2[Q12] + alf2[Q14] + alf2[Q15] + alf2[Q16] + alf2[Q17] + alf2[Q18];

                q[Q16] = alf2[Q1] - 2*alf2[Q3] + 2*alf2[Q4] - 1.5*alf2[Q5] + 1.5*alf2[Q7] - 6*alf2[Q9] + 1.5*alf2[Q10] + alf2[Q12] - alf2[Q14] + alf2[Q15] + alf2[Q16] + alf2[Q17] + alf2[Q18];

                q[Q17] = alf2[Q1] - 2*alf2[Q3] - 2*alf2[Q4] - 1.5*alf2[Q5] + 1.5*alf2[Q7] + 6*alf2[Q9] + 1.5*alf2[Q10] + alf2[Q12] - alf2[Q14] - alf2[Q15] - alf2[Q16] + alf2[Q17] + alf2[Q18];

                q[Q18] = alf2[Q1] + 2*alf2[Q3] - 2*alf2[Q4] - 1.5*alf2[Q5] + 1.5*alf2[Q7] - 6*alf2[Q9] + 1.5*alf2[Q10] - alf2[Q12] + alf2[Q14] - alf2[Q15] - alf2[Q16] + alf2[Q17] + alf2[Q18];


                Q[index(i, j, k)] = alf2;

            }
        }
    }
}



template <typename T, int QVecSize>
void ComputeUnit<T, QVecSize>::moments(){



    for (tNi i = 1;  i <= xg1; i++) {
        for (tNi j = 1; j <= yg1; j++) {

            for (tNi k = 1; k <= zg1; k++) {
                
                
                QVec<T, QVecSize> qVec = Q[index(i, j, k)];

    
                QVec<T, QVecSize> alf1 = Q[index(i, j, k)];

    
            
                
                //the first position is simply the entire mass-vector (N] summed up
                alf1[1] = qVec.q[ 1] + qVec.q[ 2] + qVec.q[ 3] + qVec.q[ 4] +
                qVec.q[ 5] + qVec.q[ 6] + qVec.q[ 7] + qVec.q[ 8] +
                qVec.q[ 9] + qVec.q[10] + qVec.q[11] + qVec.q[12] +
                qVec.q[13] + qVec.q[14] + qVec.q[15] + qVec.q[16] +
                qVec.q[17] + qVec.q[18];
                
    //                if (alf1[1]<0.00001]  printf("alf1 %f\n", alf1[1]];
                
            
                //the second position is everything with an x-component
                alf1[2] = qVec.q[ 1] - qVec.q[ 3] +
                qVec.q[ 7] - qVec.q[ 8] -
                qVec.q[ 9] + qVec.q[10] + qVec.q[11] - qVec.q[12] -
                qVec.q[13] + qVec.q[14];
                
                
                //the third position is everything with an y-component
                alf1[3] = qVec.q[ 2] - qVec.q[ 4] +
                qVec.q[ 7] + qVec.q[ 8] -
                qVec.q[ 9] - qVec.q[10] + qVec.q[15] - qVec.q[16] -
                qVec.q[17] + qVec.q[18];
                
                
                //the fourth position is everything with a z-component
                alf1[4] = qVec.q[ 5] - qVec.q[ 6] +
                qVec.q[11] + qVec.q[12] -
                qVec.q[13] - qVec.q[14] + qVec.q[15] + qVec.q[16] -
                qVec.q[17] - qVec.q[18];
                
                
                //starting from the fifth position, it gets more complicated in
                //structure, but it still follows the article by Eggels and Somers
                alf1[5] =  - qVec.q[ 2] - qVec.q[ 4] -
                qVec.q[ 5] - qVec.q[ 6] + qVec.q[ 7] + qVec.q[ 8] +
                qVec.q[ 9] + qVec.q[10] + qVec.q[11] + qVec.q[12] +
                qVec.q[13] + qVec.q[14];
                
                
                alf1[6] = qVec.q[ 7] - qVec.q[ 8] +
                qVec.q[ 9] - qVec.q[10];
                
                alf1[7] =  - qVec.q[ 1] - qVec.q[ 3] -
                qVec.q[ 5] - qVec.q[ 6] + qVec.q[ 7] + qVec.q[ 8] +
                qVec.q[ 9] + qVec.q[10] + qVec.q[15] + qVec.q[16] +
                qVec.q[17] + qVec.q[18];
                
                alf1[8] = qVec.q[11] - qVec.q[12] +
                qVec.q[13] - qVec.q[14];
                
                alf1[9] = qVec.q[15] - qVec.q[16] +
                qVec.q[17] - qVec.q[18];
                
                alf1[10] =  - qVec.q[ 1] - qVec.q[ 2] - qVec.q[ 3] - qVec.q[ 4] +
                qVec.q[11] + qVec.q[12] +
                qVec.q[13] + qVec.q[14] + qVec.q[15] + qVec.q[16] +
                qVec.q[17] + qVec.q[18];
                
                alf1[11] =  - qVec.q[ 1] + qVec.q[ 3] +
                2 * qVec.q[ 7] - 2 * qVec.q[ 8] -
                2 * qVec.q[ 9] + 2 * qVec.q[10] - qVec.q[11] + qVec.q[12] +
                qVec.q[13] - qVec.q[14];
                
                alf1[12] =  - qVec.q[ 2] + qVec.q[ 4] +
                2 * qVec.q[ 7] + 2 * qVec.q[ 8] -
                2 * qVec.q[ 9] - 2 * qVec.q[10] -
                qVec.q[15] + qVec.q[16] +
                qVec.q[17] - qVec.q[18];
                
                alf1[13] =  - 3 * qVec.q[ 1] + 3 * qVec.q[ 3] +
                3 * qVec.q[11] - 3 * qVec.q[12] -
                3 * qVec.q[13] + 3 * qVec.q[14];
                
                alf1[14] =  - 3 * qVec.q[ 2] + 3 * qVec.q[ 4] +
                3 * qVec.q[15] - 3 * qVec.q[16] -
                3 * qVec.q[17] + 3 * qVec.q[18];
                
                alf1[15] =  - 2 * qVec.q[ 5] + 2 * qVec.q[ 6] +
                qVec.q[11] + qVec.q[12] -
                qVec.q[13] - qVec.q[14] + qVec.q[15] + qVec.q[16] -
                qVec.q[17] - qVec.q[18];
                
                alf1[16] =  - 3 * qVec.q[11] - 3 * qVec.q[12] +
                3 * qVec.q[13] + 3 * qVec.q[14] + 3 * qVec.q[15] +
                3 * qVec.q[16] -
                3 * qVec.q[17] - 3 * qVec.q[18];
                
                alf1[17] = 0.5 * (qVec.q[ 1] + qVec.q[ 2] + qVec.q[ 3] + qVec.q[ 4]) -
                (qVec.q[ 5] + qVec.q[ 6] + qVec.q[ 7] + qVec.q[ 8] +
                 qVec.q[ 9] + qVec.q[10]) +
                0.5 * (qVec.q[11] + qVec.q[12] +
                       qVec.q[13] + qVec.q[14] + qVec.q[15] + qVec.q[16] +
                       qVec.q[17] + qVec.q[18]);
                
                alf1[18] = 1.5 * (qVec.q[ 1] - qVec.q[ 2] + qVec.q[ 3] - qVec.q[ 4]) -
                1.5 * (qVec.q[11] + qVec.q[12] +
                       qVec.q[13] + qVec.q[14]) +
                1.5 * (qVec.q[15] + qVec.q[16] +
                       qVec.q[17] + qVec.q[18]);
                
                
                for (int l=1; l<=18; l++){
                    Q[index(i,j,k)].q[l] = alf1[ l ];
                }
                
                
                /*
                if (i==2 && j==1 && k==1]{
                    //if (qVec.q[ 1] < 0.00001] {
                    printf("Moments %li %li %li     ", i,j,k];
                    for (int l=0; l<Q; l++]{
                        printf("% 1.4E ", qVec.q[l]];
                    }
                    printf("\n"];
                }
                */
                
            }}}//endfor i,j,k

    
    
}//end of func




