//
//  StreamingEsoteric.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
void ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Esotwist>::streamingPush() {
	//TODO
	evenStep = !evenStep;
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
void ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Esotwist>::streamingPull() {
  //TODO
  evenStep = !evenStep;
}


template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
struct AccessField<T, QVecSize, MemoryLayout, collisionType, Esotwist> {
    inline static HOST_DEVICE_GPU QVec<T, QVecSize> read(ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, Esotwist> &cu, tNi i, tNi j, tNi k) {
        if (cu.evenStep) {
            QVec<T, QVecSize> qVec;
            tNi ind0 = cu.index(i,j,k);
            qVec.q[Q01] = cu.Q[ind0].q[Q01];
            qVec.q[Q03] = cu.Q[ind0].q[Q03];
            qVec.q[Q05] = cu.Q[ind0].q[Q05];
            qVec.q[Q07] = cu.Q[ind0].q[Q07];
            qVec.q[Q09] = cu.Q[ind0].q[Q09];
            qVec.q[Q11] = cu.Q[ind0].q[Q11];
            tNi ind1 = cu.index((i==cu.xg1 ? 1 : i+1),j,k);
            qVec.q[Q02] = cu.Q[ind1].q[Q02];
            qVec.q[Q14] = cu.Q[ind1].q[Q14];
            qVec.q[Q16] = cu.Q[ind1].q[Q16];
            tNi ind3 = cu.index(i,(j==cu.yg1 ? 1 : j+1),k);
            qVec.q[Q04] = cu.Q[ind3].q[Q04];
            qVec.q[Q13] = cu.Q[ind3].q[Q13];
            qVec.q[Q18] = cu.Q[ind3].q[Q18];
            tNi ind5 = cu.index(i,j,(k==cu.zg1 ? 1 : k+1));
            qVec.q[Q06] = cu.Q[ind5].q[Q06];
            qVec.q[Q15] = cu.Q[ind5].q[Q15];
            qVec.q[Q17] = cu.Q[ind5].q[Q17];
            qVec.q[Q12] = cu.Q[cu.index(i,(j==cu.yg1 ? 1 : j+1),(k==cu.zg1 ? 1 : k+1))].q[Q12];
            qVec.q[Q10] = cu.Q[cu.index((i==cu.xg1 ? 1 : i+1),j,(k==cu.zg1 ? 1 : k+1))].q[Q10];
            qVec.q[Q08] = cu.Q[cu.index((i==cu.xg1 ? 1 : i+1),(j==cu.yg1 ? 1 : j+1),k)].q[Q08];
            return qVec;
        } else { // odd step
            QVec<T, QVecSize> qVec;
            tNi ind0 = cu.index(i,j,k);
            qVec.q[Q01] = cu.Q[ind0].q[Q02];
            qVec.q[Q03] = cu.Q[ind0].q[Q04];
            qVec.q[Q05] = cu.Q[ind0].q[Q06];
            qVec.q[Q07] = cu.Q[ind0].q[Q08];
            qVec.q[Q09] = cu.Q[ind0].q[Q10];
            qVec.q[Q11] = cu.Q[ind0].q[Q12];
            tNi ind1 = cu.index((i==cu.xg1 ? 1 : i+1),j,k);
            qVec.q[Q02] = cu.Q[ind1].q[Q01];
            qVec.q[Q14] = cu.Q[ind1].q[Q13];
            qVec.q[Q16] = cu.Q[ind1].q[Q15];
            tNi ind3 = cu.index(i,(j==cu.yg1 ? 1 : j+1),k);
            qVec.q[Q04] = cu.Q[ind3].q[Q03];
            qVec.q[Q13] = cu.Q[ind3].q[Q14];
            qVec.q[Q18] = cu.Q[ind3].q[Q17];
            tNi ind5 = cu.index(i,j,(k==cu.zg1 ? 1 : k+1));
            qVec.q[Q06] = cu.Q[ind5].q[Q05];
            qVec.q[Q15] = cu.Q[ind5].q[Q16];
            qVec.q[Q17] = cu.Q[ind5].q[Q18];
            qVec.q[Q12] = cu.Q[cu.index(i,(j==cu.yg1 ? 1 : j+1),(k==cu.zg1 ? 1 : k+1))].q[Q11];
            qVec.q[Q10] = cu.Q[cu.index((i==cu.xg1 ? 1 : i+1),j,(k==cu.zg1 ? 1 : k+1))].q[Q09];
            qVec.q[Q08] = cu.Q[cu.index((i==cu.xg1 ? 1 : i+1),(j==cu.yg1 ? 1 : j+1),k)].q[Q07];
            return qVec;
        }
    }
    inline static HOST_DEVICE_GPU void write(ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, Esotwist> &cu, QVec<T, QVecSize> &q, tNi i, tNi j, tNi k) {
        if (cu.evenStep) {
            tNi ind0 = cu.index(i,j,k);
            cu.Q[ind0].q[Q01] = q[Q02];
            cu.Q[ind0].q[Q03] = q[Q04];
            cu.Q[ind0].q[Q05] = q[Q06];
            cu.Q[ind0].q[Q07] = q[Q08];
            cu.Q[ind0].q[Q09] = q[Q10];
            cu.Q[ind0].q[Q11] = q[Q12];
            tNi ind1 = cu.index((i==cu.xg1 ? 1 : i+1),j,k);
            cu.Q[ind1].q[Q02] = q[Q01];
            cu.Q[ind1].q[Q14] = q[Q13];
            cu.Q[ind1].q[Q16] = q[Q15];
            tNi ind3 = cu.index(i,(j==cu.yg1 ? 1 : j+1),k);
            cu.Q[ind3].q[Q04] = q[Q03];
            cu.Q[ind3].q[Q13] = q[Q14];
            cu.Q[ind3].q[Q18] = q[Q17];
            tNi ind5 = cu.index(i,j,(k==cu.zg1 ? 1 : k+1));
            cu.Q[ind5].q[Q06] = q[Q05];
            cu.Q[ind5].q[Q15] = q[Q16];
            cu.Q[ind5].q[Q17] = q[Q18];
            cu.Q[cu.index(i,(j==cu.yg1 ? 1 : j+1),(k==cu.zg1 ? 1 : k+1))].q[Q12] = q[Q11];
            cu.Q[ cu.index((i==cu.xg1 ? 1 : i+1),j,(k==cu.zg1 ? 1 : k+1))].q[Q10] = q[Q09];
            cu.Q[ cu.index((i==cu.xg1 ? 1 : i+1),(j==cu.yg1 ? 1 : j+1),k)].q[Q08] = q[Q07];
        } else { // odd step
            tNi ind0 = cu.index(i,j,k);
            cu.Q[ind0].q[Q02] = q[Q02];
            cu.Q[ind0].q[Q04] = q[Q04];
            cu.Q[ind0].q[Q06] = q[Q06];
            cu.Q[ind0].q[Q08] = q[Q08];
            cu.Q[ind0].q[Q10] = q[Q10];
            cu.Q[ind0].q[Q12] = q[Q12];
            tNi ind1 = cu.index((i==cu.xg1 ? 1 : i+1),j,k);
            cu.Q[ind1].q[Q01] = q[Q01];
            cu.Q[ind1].q[Q13] = q[Q13];
            cu.Q[ind1].q[Q15] = q[Q15];
            tNi ind3 = cu.index(i,(j==cu.yg1 ? 1 : j+1),k);
            cu.Q[ind3].q[Q03] = q[Q03];
            cu.Q[ind3].q[Q14] = q[Q14];
            cu.Q[ind3].q[Q17] = q[Q17];
            tNi ind5 = cu.index(i,j,(k==cu.zg1 ? 1 : k+1));
            cu.Q[ind5].q[Q05] = q[Q05];
            cu.Q[ind5].q[Q16] = q[Q16];
            cu.Q[ind5].q[Q18] = q[Q18];
            cu.Q[cu.index(i,(j==cu.yg1 ? 1 : j+1),(k==cu.zg1 ? 1 : k+1))].q[Q11] = q[Q11];
            cu.Q[ cu.index((i==cu.xg1 ? 1 : i+1),j,(k==cu.zg1 ? 1 : k+1))].q[Q09] = q[Q09];
            cu.Q[ cu.index((i==cu.xg1 ? 1 : i+1),(j==cu.yg1 ? 1 : j+1),k)].q[Q07] = q[Q07];
        }
    }
    inline static HOST_DEVICE_GPU void writeMoments(ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, Esotwist> &cu, QVec<T, QVecSize> &q, tNi i, tNi j, tNi k) {
        if (cu.evenStep) {
            tNi ind0 = cu.index(i,j,k);
            cu.Q[ind0].q[Q01] = q[Q01];
            cu.Q[ind0].q[Q03] = q[Q03];
            cu.Q[ind0].q[Q05] = q[Q05];
            cu.Q[ind0].q[Q07] = q[Q07];
            cu.Q[ind0].q[Q09] = q[Q09];
            cu.Q[ind0].q[Q11] = q[Q11];
            tNi ind1 = cu.index((i==cu.xg1 ? 1 : i+1),j,k);
            cu.Q[ind1].q[Q02] = q[Q02];
            cu.Q[ind1].q[Q14] = q[Q14];
            cu.Q[ind1].q[Q16] = q[Q16];
            tNi ind3 = cu.index(i,(j==cu.yg1 ? 1 : j+1),k);
            cu.Q[ind3].q[Q04] = q[Q04];
            cu.Q[ind3].q[Q13] = q[Q13];
            cu.Q[ind3].q[Q18] = q[Q18];
            tNi ind5 = cu.index(i,j,(k==cu.zg1 ? 1 : k+1));
            cu.Q[ind5].q[Q06] = q[Q06];
            cu.Q[ind5].q[Q15] = q[Q15];
            cu.Q[ind5].q[Q17] = q[Q17];
            cu.Q[cu.index(i,(j==cu.yg1 ? 1 : j+1),(k==cu.zg1 ? 1 : k+1))].q[Q12] = q[Q12];
            cu.Q[cu.index((i==cu.xg1 ? 1 : i+1),j,(k==cu.zg1 ? 1 : k+1))].q[Q10] = q[Q10];
            cu.Q[cu.index((i==cu.xg1 ? 1 : i+1),(j==cu.yg1 ? 1 : j+1),k)].q[Q08] = q[Q08];
        } else { // odd step
            tNi ind0 = cu.index(i,j,k);
            cu.Q[ind0].q[Q02] = q[Q01];
            cu.Q[ind0].q[Q04] = q[Q03];
            cu.Q[ind0].q[Q06] = q[Q05];
            cu.Q[ind0].q[Q08] = q[Q07];
            cu.Q[ind0].q[Q10] = q[Q09];
            cu.Q[ind0].q[Q12] = q[Q11];
            tNi ind1 = cu.index((i==cu.xg1 ? 1 : i+1),j,k);
            cu.Q[ind1].q[Q01] = q[Q02];
            cu.Q[ind1].q[Q13] = q[Q14];
            cu.Q[ind1].q[Q15] = q[Q16];
            tNi ind3 = cu.index(i,(j==cu.yg1 ? 1 : j+1),k);
            cu.Q[ind3].q[Q03] = q[Q04];
            cu.Q[ind3].q[Q14] = q[Q13];
            cu.Q[ind3].q[Q17] = q[Q18];
            tNi ind5 = cu.index(i,j,(k==cu.zg1 ? 1 : k+1));
            cu.Q[ind5].q[Q05] = q[Q06];
            cu.Q[ind5].q[Q16] = q[Q15];
            cu.Q[ind5].q[Q18] = q[Q17];
            cu.Q[cu.index(i,(j==cu.yg1 ? 1 : j+1),(k==cu.zg1 ? 1 : k+1))].q[Q11] = q[Q12];
            cu.Q[cu.index((i==cu.xg1 ? 1 : i+1),j,(k==cu.zg1 ? 1 : k+1))].q[Q09] = q[Q10];
            cu.Q[cu.index((i==cu.xg1 ? 1 : i+1),(j==cu.yg1 ? 1 : j+1),k)].q[Q07] = q[Q08];
        }
    }
};

