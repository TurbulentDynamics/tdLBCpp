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
    using AF = AccessField<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>;

    //kinematic viscosity.
    T b = 1.0 / (1.0 + 6 * flow.nu);
    T c = 1.0 - 6 * flow.nu;


    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){


                Force<T> f = F[index(i,j,k)];


                QVec<T, QVecSize> m = AF::read(*this, i, j, k);


                Velocity<T> u = m.velocity(f);

                QVec<T, QVecSize> alpha;


                if (flow.useLES == 1){
                    T fct = 3.0 / (m.q[M01] * (1.0 + 6.0 * (Nu[index(i,j,k)] + flow.nu)));

                    //calculating the derivatives for x, y and z
                    T dudx = fct * ((m.q[M02] + 0.5 * F[index(i,j,k)].x * u.x - m.q[M05]));
                    T dvdy = fct * ((m.q[M03] + 0.5 * F[index(i,j,k)].y * u.y - m.q[M07]));
                    T dwdz = fct * ((m.q[M04] + 0.5 * F[index(i,j,k)].z * u.z - m.q[M10]));

                    T divv = dudx + dvdy + dwdz;


                    //calculating the cross terms, used for the shear matrix
                    T dudypdvdx = 2 * fct * ((m.q[M03]) + 0.5 * F[index(i,j,k)].y * u.x - m.q[M06]);
                    T dudzpdwdx = 2 * fct * ((m.q[M04]) + 0.5 * F[index(i,j,k)].z * u.x - m.q[M08]);
                    T dvdzpdwdy = 2 * fct * ((m.q[M04]) + 0.5 * F[index(i,j,k)].z * u.y - m.q[M09]);


                    //calculating sh (the resolved deformation rate, S^2)
                    T sh = 2 * dudx*dudx + 2 * dvdy*dvdy + 2 * dwdz*dwdz + dudypdvdx*dudypdvdx + dudzpdwdx*dudzpdwdx + dvdzpdwdy*dvdzpdwdy - (2.0 / 3.0) * divv*divv;


                    //calculating eddy viscosity:
                    //nu_t = (lambda_mix)^2 * sqrt(S^2)     (Smagorinsky)
                    Nu[index(i,j,k)] = flow.cs0 * flow.cs0 * sqrt(fabs(sh));



                    // Viscosity is adjusted only for LES, because LES uses a
                    // subgrid-adjustment model for turbulence that's too small to
                    // be captured in the regular cells. This adjustment is
                    // performed by adding the eddy viscosity to the viscosity.
                    // This model is called the Smagorinsky model, however this
                    // implementation is slightly different, as explained by
                    // Somers (1993) -> low strain rates do not excite the
                    // eddy viscosity.

                    T nut = Nu[index(i,j,k)] + flow.nu;
                    b = 1.0 / (1.0 + 6 * nut);
                    c = 1.0 - 6 * nut;

                }//end of LES



                //0th order term
                alpha[M01] = m[M01];


                //1st order term
                alpha[M02] = m[M02] + f.x;
                alpha[M03] = m[M03] + f.y;
                alpha[M04] = m[M04] + f.z;

                //2nd order terms
                alpha[M05] = (2.0 * (m[M02] + 0.5 * f.x) * u.x - m[M05]*c)*b;
                alpha[M06] = (2.0 * (m[M02] + 0.5 * f.x) * u.y - m[M06]*c)*b;
                alpha[M07] = (2.0 * (m[M03] + 0.5 * f.y) * u.y - m[M07]*c)*b;

                alpha[M08] = (2.0 * (m[M02] + 0.5 * f.x) * u.z - m[M08]*c)*b;
                alpha[M09] = (2.0 * (m[M03] + 0.5 * f.y) * u.z - m[M09]*c)*b;
                alpha[M10] = (2.0 * (m[M04] + 0.5 * f.z) * u.z - m[M10]*c)*b;

                //3rd order terms
                alpha[M11] =  -flow.g3 * m[M11];
                alpha[M12] =  -flow.g3 * m[M12];
                alpha[M13] =  -flow.g3 * m[M13];
                alpha[M14] =  -flow.g3 * m[M14];
                alpha[M15] =  -flow.g3 * m[M15];
                alpha[M16] =  -flow.g3 * m[M16];

                //4th order terms
                alpha[M17] = 0.0;
                alpha[M18] = 0.0;


                // Start of invMoments, which is responsible for determining
                // the N-field (x) from alpha+ (alpha). It does this by using eq.
                // 12 in the article by Eggels and Somers (1995), which means
                // it's using the "filter matrix E" (not really present in the
                // code as matrix, but it's where the coefficients come from).

                for (int l=0;  l<QVecSize; l++) {
                    alpha[l] /= 24.0;
                }


                m[Q01] = 2*alpha[M01] + 4*alpha[M02] + 3*alpha[M05] - 3*alpha[M07] - 3*alpha[M10] - 2*alpha[M11] - 2*alpha[M13] + 2*alpha[M17] + 2*alpha[M18];

                m[Q02] = 2*alpha[M01] - 4*alpha[M02] + 3*alpha[M05] - 3*alpha[M07] - 3*alpha[M10] + 2*alpha[M11] + 2*alpha[M13] + 2*alpha[M17] + 2*alpha[M18];

                m[Q03] = 2*alpha[M01] + 4*alpha[M03] - 3*alpha[M05] + 3*alpha[M07] - 3*alpha[M10] - 2*alpha[M12] - 2*alpha[M14] + 2*alpha[M17] - 2*alpha[M18];

                m[Q04] = 2*alpha[M01] - 4*alpha[M03] - 3*alpha[M05] + 3*alpha[M07] - 3*alpha[M10] + 2*alpha[M12] + 2*alpha[M14] + 2*alpha[M17] - 2*alpha[M18];

                m[Q05] = 2*alpha[M01] + 4*alpha[M04] - 3*alpha[M05] - 3*alpha[M07] + 3*alpha[M10] - 4*alpha[M15] - 4*alpha[M17];

                m[Q06] = 2*alpha[M01] - 4*alpha[M04] - 3*alpha[M05] - 3*alpha[M07] + 3*alpha[M10] + 4*alpha[M15] - 4*alpha[M17];

                m[Q07] = alpha[M01] + 2*alpha[M02] + 2*alpha[M03] + 1.5*alpha[M05] + 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] + 2*alpha[M11] + 2*alpha[M12] - 2*alpha[M17];

                m[M14] = alpha[M01] - 2*alpha[M02] + 2*alpha[M03] + 1.5*alpha[M05] - 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] - 2*alpha[M11] + 2*alpha[M12] - 2*alpha[M17];

                m[M08] = alpha[M01] - 2*alpha[M02] - 2*alpha[M03] + 1.5*alpha[M05] + 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] - 2*alpha[M11] - 2*alpha[M12] - 2*alpha[M17];

                m[M13] = alpha[M01] + 2*alpha[M02] - 2*alpha[M03] + 1.5*alpha[M05] - 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] + 2*alpha[M11] - 2*alpha[M12] - 2*alpha[M17];

                m[M09] = alpha[M01] + 2*alpha[M02] + 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] + 6*alpha[M08] + 1.5*alpha[M10] - alpha[M11] + alpha[M13] + alpha[M15] - alpha[M16] + alpha[M17] - alpha[M18];

                m[M16] = alpha[M01] - 2*alpha[M02] + 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] - 6*alpha[M08] + 1.5*alpha[M10] + alpha[M11] - alpha[M13] + alpha[M15] - alpha[M16] + alpha[M17] - alpha[M18];

                m[M10] = alpha[M01] - 2*alpha[M02] - 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] + 6*alpha[M08] + 1.5*alpha[M10] + alpha[M11] - alpha[M13] - alpha[M15] + alpha[M16] + alpha[M17] - alpha[M18];

                m[M15] = alpha[M01] + 2*alpha[M02] - 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] - 6*alpha[M08] + 1.5*alpha[M10] - alpha[M11] + alpha[M13] - alpha[M15] + alpha[M16] + alpha[M17] - alpha[M18];

                m[M11] = alpha[M01] + 2*alpha[M03] + 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] + 6*alpha[M09] + 1.5*alpha[M10] - alpha[M12] + alpha[M14] + alpha[M15] + alpha[M16] + alpha[M17] + alpha[M18];

                m[M18] = alpha[M01] - 2*alpha[M03] + 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] - 6*alpha[M09] + 1.5*alpha[M10] + alpha[M12] - alpha[M14] + alpha[M15] + alpha[M16] + alpha[M17] + alpha[M18];

                m[M12] = alpha[M01] - 2*alpha[M03] - 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] + 6*alpha[M09] + 1.5*alpha[M10] + alpha[M12] - alpha[M14] - alpha[M15] - alpha[M16] + alpha[M17] + alpha[M18];

                m[M17] = alpha[M01] + 2*alpha[M03] - 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] - 6*alpha[M09] + 1.5*alpha[M10] - alpha[M12] + alpha[M14] - alpha[M15] - alpha[M16] + alpha[M17] + alpha[M18];




                AF::write(*this, m, i, j, k);

            }
        }
    }
}



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, streamingType>::moments(){

    using AF = AccessField<T, QVecSize, MemoryLayout, collisionType, streamingType>;

    for (tNi i = 1;  i <= xg1; i++) {
        for (tNi j = 1; j <= yg1; j++) {
            for (tNi k = 1; k <= zg1; k++) {


                QVec<T, QVecSize> q = AF::read(*this, i, j, k);


                QVec<T, QVecSize> m;

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

                AF::writeMoments(*this, m, i, j, k);

            }
        }
    }



}//end of func




