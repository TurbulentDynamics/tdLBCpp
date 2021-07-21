//
//  StreamingEsoteric.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
void ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Esotwist>::streaming() {
	//TODO
	evenStep = !evenStep;
}


template<typename T, int QVecSize, MemoryLayoutType MemoryLayout>
struct AccessField<T, QVecSize, MemoryLayout, Esotwist> {
    inline static QVec<T, QVecSize> read(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, tNi i, tNi j, tNi k) {
        if (cu.evenStep) {
            QVec<T, QVecSize> qVec;
            tNi ind0 = cu.index(i,j,k);
            qVec.q[ Q1] = cu.Q[ind0].q[ Q1];
            qVec.q[ Q3] = cu.Q[ind0].q[ Q3];
            qVec.q[ Q5] = cu.Q[ind0].q[ Q5];
            qVec.q[ Q7] = cu.Q[ind0].q[ Q7];
            qVec.q[ Q9] = cu.Q[ind0].q[ Q9];
            qVec.q[Q11] = cu.Q[ind0].q[Q11];
            tNi ind1 = cu.dirnQ1(i,j,k);
            qVec.q[ Q2] = cu.Q[ind1].q[ Q2];
            qVec.q[Q14] = cu.Q[ind1].q[Q14];
            qVec.q[Q16] = cu.Q[ind1].q[Q16];
            tNi ind3 = cu.dirnQ3(i,j,k);
            qVec.q[ Q4] = cu.Q[ind3].q[ Q4];
            qVec.q[Q13] = cu.Q[ind3].q[Q13];
            qVec.q[Q18] = cu.Q[ind3].q[Q18];
            tNi ind5 = cu.dirnQ5(i,j,k);
            qVec.q[ Q6] = cu.Q[ind5].q[ Q6];
            qVec.q[Q15] = cu.Q[ind5].q[Q15];
            qVec.q[Q17] = cu.Q[ind5].q[Q17];
            qVec.q[Q12] = cu.Q[cu.dirnQ11(i,j,k)].q[Q12];
            qVec.q[Q10] = cu.Q[ cu.dirnQ9(i,j,k)].q[Q10];
            qVec.q[ Q8] = cu.Q[ cu.dirnQ7(i,j,k)].q[ Q8];
            return qVec;
        } else { // odd step
            QVec<T, QVecSize> qVec;
            tNi ind0 = cu.index(i,j,k);
            qVec.q[ Q1] = cu.Q[ind0].q[ Q2];
            qVec.q[ Q3] = cu.Q[ind0].q[ Q4];
            qVec.q[ Q5] = cu.Q[ind0].q[ Q6];
            qVec.q[ Q7] = cu.Q[ind0].q[ Q8];
            qVec.q[ Q9] = cu.Q[ind0].q[Q10];
            qVec.q[Q11] = cu.Q[ind0].q[Q12];
            tNi ind1 = cu.dirnQ1(i,j,k);
            qVec.q[ Q2] = cu.Q[ind1].q[ Q1];
            qVec.q[Q14] = cu.Q[ind1].q[Q13];
            qVec.q[Q16] = cu.Q[ind1].q[Q15];
            tNi ind3 = cu.dirnQ3(i,j,k);
            qVec.q[ Q4] = cu.Q[ind3].q[ Q3];
            qVec.q[Q13] = cu.Q[ind3].q[Q14];
            qVec.q[Q18] = cu.Q[ind3].q[Q17];
            tNi ind5 = cu.dirnQ5(i,j,k);
            qVec.q[ Q6] = cu.Q[ind5].q[ Q5];
            qVec.q[Q15] = cu.Q[ind5].q[Q16];
            qVec.q[Q17] = cu.Q[ind5].q[Q18];
            qVec.q[Q12] = cu.Q[cu.dirnQ11(i,j,k)].q[Q11];
            qVec.q[Q10] = cu.Q[ cu.dirnQ9(i,j,k)].q[ Q9];
            qVec.q[ Q8] = cu.Q[ cu.dirnQ7(i,j,k)].q[ Q7];
            return qVec;
        }
    }
    inline static void write(ComputeUnitBase<T, QVecSize, MemoryLayout> &cu, QVec<T, QVecSize> &q, tNi i, tNi j, tNi k) {
        if (cu.evenStep) {
            tNi ind0 = cu.index(i,j,k);
            cu.Q[ind0].q[ Q1] = q[ Q2]; 
            cu.Q[ind0].q[ Q3] = q[ Q4];
            cu.Q[ind0].q[ Q5] = q[ Q6];
            cu.Q[ind0].q[ Q7] = q[ Q8];
            cu.Q[ind0].q[ Q9] = q[Q10];
            cu.Q[ind0].q[Q11] = q[Q12];
            tNi ind1 = cu.dirnQ1(i,j,k);
            cu.Q[ind1].q[ Q2] = q[ Q1];
            cu.Q[ind1].q[Q14] = q[Q13];
            cu.Q[ind1].q[Q16] = q[Q15];
            tNi ind3 = cu.dirnQ3(i,j,k);
            cu.Q[ind3].q[ Q4] = q[ Q3];
            cu.Q[ind3].q[Q13] = q[Q14];
            cu.Q[ind3].q[Q18] = q[Q17];
            tNi ind5 = cu.dirnQ5(i,j,k);
            cu.Q[ind5].q[ Q6] = q[ Q5];
            cu.Q[ind5].q[Q15] = q[Q16];
            cu.Q[ind5].q[Q17] = q[Q18];
            cu.Q[cu.dirnQ11(i,j,k)].q[Q12] = q[Q11];
            cu.Q[ cu.dirnQ9(i,j,k)].q[Q10] = q[ Q9];
            cu.Q[ cu.dirnQ7(i,j,k)].q[ Q8] = q[ Q7];
        } else { // odd step
            tNi ind0 = cu.index(i,j,k);
            cu.Q[ind0].q[ Q2] = q[ Q2];
            cu.Q[ind0].q[ Q4] = q[ Q4];
            cu.Q[ind0].q[ Q6] = q[ Q6];
            cu.Q[ind0].q[ Q8] = q[ Q8];
            cu.Q[ind0].q[Q10] = q[Q10];
            cu.Q[ind0].q[Q12] = q[Q12];
            tNi ind1 = cu.dirnQ1(i,j,k);
            cu.Q[ind1].q[ Q1] = q[ Q1];
            cu.Q[ind1].q[Q13] = q[Q13];
            cu.Q[ind1].q[Q15] = q[Q15];
            tNi ind3 = cu.dirnQ3(i,j,k);
            cu.Q[ind3].q[ Q3] = q[ Q3];
            cu.Q[ind3].q[Q14] = q[Q14];
            cu.Q[ind3].q[Q17] = q[Q17];
            tNi ind5 = cu.dirnQ5(i,j,k);
            cu.Q[ind5].q[ Q5] = q[ Q5];
            cu.Q[ind5].q[Q16] = q[Q16];
            cu.Q[ind5].q[Q18] = q[Q18];
            cu.Q[cu.dirnQ11(i,j,k)].q[Q11] = q[Q11];
            cu.Q[ cu.dirnQ9(i,j,k)].q[ Q9] = q[ Q9];
            cu.Q[ cu.dirnQ7(i,j,k)].q[ Q7] = q[ Q7];
        }
    }
};

