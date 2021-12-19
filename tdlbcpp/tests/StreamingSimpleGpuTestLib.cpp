#include "Header.h"

HOST_DEVICE_GPU unsigned long fabs(unsigned long v) {
    return v;
}

HOST_DEVICE_GPU unsigned long log(unsigned long v) {
    return v;
}

HOST_DEVICE_GPU unsigned long sqrt(unsigned long v) {
    return v;
}

#include "ComputeUnit.h"

#include "tdlbcpp/tests/Params/ParamsCommon.hpp"

void createGpuUnitsExecutePush(unsigned long *q, unsigned long *qLijk, ComputeUnitParams cuParams, FlowParams<unsigned long> flow, DiskOutputTree diskOutputTree) {
    ComputeUnit<unsigned long, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple, GPU> lb2(cuParams, flow, diskOutputTree);
    lb2.initialise(0);
    ComputeUnit<unsigned long, QLen::D3Q19, MemoryLayoutLIJK, EgglesSomers, Simple, GPU> lb2lijk(cuParams, flow, diskOutputTree);
    lb2lijk.initialise(0);
    ParamsCommon::fillForTestGpu(lb2);
    lb2.streamingPush();
    // move memory from gpu to prepare for 
    checkCudaErrors(cudaMemcpy(q, lb2.Q.q, sizeof(unsigned long) * lb2.Q.qSize, cudaMemcpyDeviceToHost));

    ParamsCommon::fillForTestGpu(lb2lijk);
    lb2lijk.streamingPush();
    checkCudaErrors(cudaMemcpy(qLijk, lb2lijk.Q.q, sizeof(unsigned long) * lb2lijk.Q.qSize, cudaMemcpyDeviceToHost));
}