//
//  CollisionEgglesSomers.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"







template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
void ComputeUnitCollision<T, QVecSize, MemoryLayout, EgglesSomersLES, streamingType>::collision(){

    //    alpha[ 5] = (2.0*(q[2] + 0.5*f.x)*u.x - q[ 5]*c)*b * Nu.getNu(i,j,k);

}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
void ComputeUnitCollision<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>::collision(){
    using AF = AccessField<T, QVecSize, MemoryLayout, streamingType>;

    //kinematic viscosity.
    T b = 1.0 / (1.0 + 6 * flow.nu);
    T c = 1.0 - 6 * flow.nu;


    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){


                Force<T> f = F[index(i,j,k)];


                //TODO Change this to m, but write to q, notation only
                QVec<T, QVecSize> q = AF::read(*this, i, j, k);


                Velocity<T> u = q.velocity(f);

                QVec<T, QVecSize> alpha;




//                if (flow.useLES == 1){
//                    T fct = 3.0 / (q.q[M01] * (1.0 + 6.0 * (Nu[index(i,j,k)] + flow.nu)));
//
//                    //calculating the derivatives for x, y and z
//                    T dudx = fct * ((q.q[M02] + 0.5 * F[index(i,j,k)].x * u.x - q.q[M05]));
//                    T dvdy = fct * ((q.q[M03] + 0.5 * F[index(i,j,k)].y * u.y - q.q[M07]));
//                    T dwdz = fct * ((q.q[M04] + 0.5 * F[index(i,j,k)].z * u.z - q.q[M10]));
//
//                    T divv = dudx + dvdy + dwdz;
//
//
//                    //calculating the cross terms, used for the shear matrix
//                    T dudypdvdx = 2 * fct * ((q.q[M03]) + 0.5 * F[index(i,j,k)].y * u.x - q.q[M06]);
//                    T dudzpdwdx = 2 * fct * ((q.q[M04]) + 0.5 * F[index(i,j,k)].z * u.x - q.q[M08]);
//                    T dvdzpdwdy = 2 * fct * ((q.q[M04]) + 0.5 * F[index(i,j,k)].z * u.y - q.q[M09]);
//
//
//                    //calculating sh (the resolved deformation rate, S^2)
//                    T sh = 2 * pow(dudx,2) + 2 * pow(dvdy,2) + 2 * pow(dwdz,2) + pow(dudypdvdx,2) + pow(dudzpdwdx,2) + pow(dvdzpdwdy,2) - (2.0/3.0) * pow(divv,2);
//
//
//                    //calculating eddy viscosity:
//                    //nu_t = (lambda_mix)^2 * sqrt(S^2)     (Smagorinsky)
//                    Nu[index(i,j,k)] = flow.cs0 * flow.cs0 * sqrt(fabs(sh));
//
//
//
//                    /* Viscosity is adjusted only for LES, because LES uses a
//                     * subgrid-adjustment model for turbulence that's too small to
//                     * be captured in the regular cells. This adjustment is
//                     * performed by adding the eddy viscosity to the viscosity.
//                     * This model is called the Smagorinsky model, however this
//                     * implementation is slightly different, as explained by
//                     * Somers (1993) -> low strain rates do not excite the
//                     * eddy viscosity.
//                     */
//                    T nut = Nu[index(i,j,k)] + flow.nu;
//                    b = 1.0 / (1.0 + 6 * nut);
//                    c = 1.0 - 6 * nut;
//
//                }//end of LES


                
                //0th order term
                alpha[M01] = q[M01];


                //1st order term
                alpha[M02] = q[M02] + f.x;
                alpha[M03] = q[M03] + f.y;
                alpha[M04] = q[M04] + f.z;

                //2nd order terms
                alpha[Q05] = (2.0 * (q[M02] + 0.5 * f.x) * u.x - q[Q05]*c)*b;
                alpha[Q06] = (2.0 * (q[M02] + 0.5 * f.x) * u.y - q[Q06]*c)*b;
                alpha[Q07] = (2.0 * (q[M03] + 0.5 * f.y) * u.y - q[Q07]*c)*b;

                alpha[Q08] = (2.0 * (q[M02] + 0.5 * f.x) * u.z - q[M08]*c)*b;
                alpha[Q09] = (2.0 * (q[M03] + 0.5 * f.y) * u.z - q[M09]*c)*b;
                alpha[Q10] = (2.0 * (q[M04] + 0.5 * f.z) * u.z - q[M10]*c)*b;

                //3rd order terms
                alpha[Q11] =  -flow.g3 * q[Q11];
                alpha[Q12] =  -flow.g3 * q[Q12];
                alpha[Q13] =  -flow.g3 * q[Q13];
                alpha[Q14] =  -flow.g3 * q[Q14];
                alpha[Q15] =  -flow.g3 * q[Q15];
                alpha[Q16] =  -flow.g3 * q[Q16];

                //4th order terms
                alpha[Q17] = 0.0;
                alpha[Q18] = 0.0;


                // Start of invMoments, which is responsible for determining
                // the N-field (x) from alpha+ (alpha). It does this by using eq.
                // 12 in the article by Eggels and Somers (1995), which means
                // it's using the "filter matrix E" (not really present in the
                // code as matrix, but it's where the coefficients come from).

                for (int l=0;  l<QVecSize; l++) {
                    alpha[l] /= 24.0;
                }


                q[Q01] = 2*alpha[M01] + 4*alpha[M02] + 3*alpha[Q05] - 3*alpha[Q07] - 3*alpha[Q10] - 2*alpha[Q11] - 2*alpha[Q13] + 2*alpha[Q17] + 2*alpha[Q18];

                q[Q02] = 2*alpha[M01] - 4*alpha[M02] + 3*alpha[Q05] - 3*alpha[Q07] - 3*alpha[Q10] + 2*alpha[Q11] + 2*alpha[Q13] + 2*alpha[Q17] + 2*alpha[Q18];

                q[Q03] = 2*alpha[M01] + 4*alpha[M03] - 3*alpha[Q05] + 3*alpha[Q07] - 3*alpha[Q10] - 2*alpha[Q12] - 2*alpha[Q14] + 2*alpha[Q17] - 2*alpha[Q18];

                q[Q04] = 2*alpha[M01] - 4*alpha[M03] - 3*alpha[Q05] + 3*alpha[Q07] - 3*alpha[Q10] + 2*alpha[Q12] + 2*alpha[Q14] + 2*alpha[Q17] - 2*alpha[Q18];

                q[Q05] = 2*alpha[M01] + 4*alpha[M04] - 3*alpha[Q05] - 3*alpha[Q07] + 3*alpha[Q10] - 4*alpha[Q15] - 4*alpha[Q17];

                q[Q06] = 2*alpha[M01] - 4*alpha[M04] - 3*alpha[Q05] - 3*alpha[Q07] + 3*alpha[Q10] + 4*alpha[Q15] - 4*alpha[Q17];

                q[Q07] = alpha[M01] + 2*alpha[M02] + 2*alpha[M03] + 1.5*alpha[Q05] + 6*alpha[Q06] + 1.5*alpha[Q07] - 1.5*alpha[Q10] + 2*alpha[Q11] + 2*alpha[Q12] - 2*alpha[Q17];

                q[M14] = alpha[M01] - 2*alpha[M02] + 2*alpha[M03] + 1.5*alpha[Q05] - 6*alpha[Q06] + 1.5*alpha[Q07] - 1.5*alpha[Q10] - 2*alpha[Q11] + 2*alpha[Q12] - 2*alpha[Q17];

                q[M08] = alpha[M01] - 2*alpha[M02] - 2*alpha[M03] + 1.5*alpha[Q05] + 6*alpha[Q06] + 1.5*alpha[Q07] - 1.5*alpha[Q10] - 2*alpha[Q11] - 2*alpha[Q12] - 2*alpha[Q17];

                q[M13] = alpha[M01] + 2*alpha[M02] - 2*alpha[M03] + 1.5*alpha[Q05] - 6*alpha[Q06] + 1.5*alpha[Q07] - 1.5*alpha[Q10] + 2*alpha[Q11] - 2*alpha[Q12] - 2*alpha[Q17];

                q[M09] = alpha[M01] + 2*alpha[M02] + 2*alpha[M04] + 1.5*alpha[Q05] - 1.5*alpha[Q07] + 6*alpha[Q08] + 1.5*alpha[Q10] - alpha[Q11] + alpha[Q13] + alpha[Q15] - alpha[Q16] + alpha[Q17] - alpha[Q18];

                q[M16] = alpha[M01] - 2*alpha[M02] + 2*alpha[M04] + 1.5*alpha[Q05] - 1.5*alpha[Q07] - 6*alpha[Q08] + 1.5*alpha[Q10] + alpha[Q11] - alpha[Q13] + alpha[Q15] - alpha[Q16] + alpha[Q17] - alpha[Q18];

                q[M10] = alpha[M01] - 2*alpha[M02] - 2*alpha[M04] + 1.5*alpha[Q05] - 1.5*alpha[Q07] + 6*alpha[Q08] + 1.5*alpha[Q10] + alpha[Q11] - alpha[Q13] - alpha[Q15] + alpha[Q16] + alpha[Q17] - alpha[Q18];

                q[M15] = alpha[M01] + 2*alpha[M02] - 2*alpha[M04] + 1.5*alpha[Q05] - 1.5*alpha[Q07] - 6*alpha[Q08] + 1.5*alpha[Q10] - alpha[Q11] + alpha[Q13] - alpha[Q15] + alpha[Q16] + alpha[Q17] - alpha[Q18];

                q[M11] = alpha[M01] + 2*alpha[M03] + 2*alpha[M04] - 1.5*alpha[Q05] + 1.5*alpha[Q07] + 6*alpha[Q09] + 1.5*alpha[Q10] - alpha[Q12] + alpha[Q14] + alpha[Q15] + alpha[Q16] + alpha[Q17] + alpha[Q18];

                q[M18] = alpha[M01] - 2*alpha[M03] + 2*alpha[M04] - 1.5*alpha[Q05] + 1.5*alpha[Q07] - 6*alpha[Q09] + 1.5*alpha[Q10] + alpha[Q12] - alpha[Q14] + alpha[Q15] + alpha[Q16] + alpha[Q17] + alpha[Q18];

                q[M12] = alpha[M01] - 2*alpha[M03] - 2*alpha[M04] - 1.5*alpha[Q05] + 1.5*alpha[Q07] + 6*alpha[Q09] + 1.5*alpha[Q10] + alpha[Q12] - alpha[Q14] - alpha[Q15] - alpha[Q16] + alpha[Q17] + alpha[Q18];

                q[M17] = alpha[M01] + 2*alpha[M03] - 2*alpha[M04] - 1.5*alpha[Q05] + 1.5*alpha[Q07] - 6*alpha[Q09] + 1.5*alpha[Q10] - alpha[Q12] + alpha[Q14] - alpha[Q15] - alpha[Q16] + alpha[Q17] + alpha[Q18];




                AF::write(*this, q, i, j, k);

            }
        }
    }
}



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
void ComputeUnitCollision<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>::moments(){

    using QVecAcc = QVecAccess<T, QVecSize, MemoryLayout>;

    for (tNi i = 1;  i <= xg1; i++) {
        for (tNi j = 1; j <= yg1; j++) {
            for (tNi k = 1; k <= zg1; k++) {


                QVecAcc q = Q[index(i, j, k)];


                QVec<T, QVecSize> m = Q[index(i, j, k)];


                //the first position is simply the entire mass-vector (Q summed up)
                m[M01] = q.q[Q01] + q.q[Q03] + q.q[Q02] + q.q[Q04] + q.q[Q05] + q.q[Q06] + q.q[Q07] + q.q[Q14] + q.q[Q08] + q.q[Q13] + q.q[Q09] + q.q[Q16] + q.q[Q10] + q.q[Q15] + q.q[Q11] + q.q[Q18] + q.q[Q12] + q.q[Q17];


                //the second position is everything with an x-component
                m[M02] = q.q[Q01] - q.q[Q02] + q.q[Q07] - q.q[Q14] - q.q[Q08] + q.q[Q13] + q.q[Q09] - q.q[Q16] - q.q[Q10] + q.q[Q15];


                //the third position is everything with an y-component
                m[M03] = q.q[Q03] - q.q[Q04] + q.q[Q07] + q.q[Q14] - q.q[Q08] - q.q[Q13] + q.q[Q11] - q.q[Q18] - q.q[Q12] + q.q[Q17];


                //the fourth position is everything with a z-component
                m[M04] = q.q[Q05] - q.q[Q06] + q.q[Q09] + q.q[Q16] - q.q[Q10] - q.q[Q15] + q.q[Q11] + q.q[Q18] - q.q[Q12] - q.q[Q17];


                //starting from the fifth position, it gets more complicated in
                //structure, but it still follows the article by Eggels and Somers
                m[M05] =  - q.q[Q03] - q.q[Q04] - q.q[Q05] - q.q[Q06] + q.q[Q07] + q.q[Q14] + q.q[Q08] + q.q[Q13] + q.q[Q09] + q.q[Q16] + q.q[Q10] + q.q[Q15];


                m[M06] = q.q[Q07] - q.q[Q14] + q.q[Q08] - q.q[Q13];

                m[M07] =  - q.q[Q01] - q.q[Q02] - q.q[Q05] - q.q[Q06] + q.q[Q07] + q.q[Q14] + q.q[Q08] + q.q[Q13] + q.q[Q11] + q.q[Q18] + q.q[Q12] + q.q[Q17];

                m[M08] = q.q[Q09] - q.q[Q16] + q.q[Q10] - q.q[Q15];

                m[M09] = q.q[Q11] - q.q[Q18] + q.q[Q12] - q.q[Q17];

                m[M10] =  - q.q[Q01] - q.q[Q03] - q.q[Q02] - q.q[Q04] + q.q[Q09] + q.q[Q16] + q.q[Q10] + q.q[Q15] + q.q[Q11] + q.q[Q18] + q.q[Q12] + q.q[Q17];

                m[M11] =  - q.q[Q01] + q.q[Q02] + 2*q.q[Q07] - 2*q.q[Q14] - 2*q.q[Q08] + 2*q.q[Q13] - q.q[Q09] + q.q[Q16] + q.q[Q10] - q.q[Q15];

                m[M12] =  - q.q[Q03] + q.q[Q04] + 2*q.q[Q07] + 2*q.q[Q14] - 2*q.q[Q08] - 2*q.q[Q13] - q.q[Q11] + q.q[Q18] + q.q[Q12] - q.q[Q17];

                m[M13] =  - 3*q.q[Q01] + 3*q.q[Q02] + 3*q.q[Q09] - 3* q.q[Q16] - 3*q.q[Q10] + 3*q.q[Q15];

                m[M14] =  - 3*q.q[Q03] + 3*q.q[Q04] + 3*q.q[Q11] - 3*q.q[Q18] - 3*q.q[Q12] + 3*q.q[Q17];

                m[M15] =  - 2*q.q[Q05] + 2*q.q[Q06] + q.q[Q09] + q.q[Q16] - q.q[Q10] - q.q[Q15] + q.q[Q11] + q.q[Q18] - q.q[Q12] - q.q[Q17];

                m[M16] =  - 3*q.q[Q09] - 3*q.q[Q16] + 3*q.q[Q10] + 3*q.q[Q15] + 3*q.q[Q11] + 3*q.q[Q18] - 3*q.q[Q12] - 3*q.q[Q17];

                m[M17] = 0.5*q.q[Q01] + 0.5*q.q[Q03] + 0.5*q.q[Q02] + 0.5*q.q[Q04] - q.q[Q05] - q.q[Q06] - q.q[Q07] - q.q[Q14] - q.q[Q08] - q.q[Q13] + 0.5*q.q[Q09] + 0.5*q.q[Q16] + 0.5*q.q[Q10] + 0.5*q.q[Q15] + 0.5*q.q[Q11] + 0.5*q.q[Q18] + 0.5*q.q[Q12] + 0.5*q.q[Q17];

                m[M18] = 1.5*q.q[Q01] - 1.5*q.q[Q03] + 1.5*q.q[Q02] - 1.5*q.q[Q04] - 1.5*q.q[Q09] - 1.5* q.q[Q16] - 1.5* q.q[Q10] - 1.5* q.q[Q15] + 1.5*q.q[Q11] + 1.5*q.q[Q18] + 1.5*q.q[Q12] + 1.5*q.q[Q17];




                for (int l=0; l<QVecSize; l++){
                    Q[index(i,j,k)].q[l] = m[ l ];
                }


                /*
                 if (i==2 && j==1 && k==1]{
                 //if (qVec.q[ 1] < 0.00001] {
                 printf("Moments %li %li %li     ", i,j,k];
                 for (int l=0; l<N; l++]{
                 printf("% 1.4E ", qVec.q[l]];
                 }
                 printf("\n"];
                 }
                 */

            }}}//endfor i,j,k



}//end of func




