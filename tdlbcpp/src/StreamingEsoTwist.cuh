//
//  StreamingSimple.cuh
//  tdLBcpp
//
//  Created by Niall Ó Broin on 2021/09/03.
//

#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "ComputeUnit.h"

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType>
__global__ void streamingPush(ComputeUnitStreaming<T, QVecSize, MemoryLayout, collisionType, Esotwist> &cu) {
	//TODO
	cu.evenStep = !cu.evenStep;
}















