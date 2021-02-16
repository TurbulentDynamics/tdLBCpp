//
//  ComputeGroup.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include <stdio.h>
#include <vector>

#ifdef DEBUG
#include <iostream>
#endif

#include "Header.h"
#include "FlowParams.hpp"
#include "QVec.hpp"

#if WITH_GPU == 1
    #include <cuda_runtime.h>
    #include "helper_cuda.h"
#endif

template <typename T, int QVecSize>
class ComputeUnit {
public:

    tNi idi, idj, idk;
    tNi x, y, z;
    tNi xg, yg, zg;
    tNi ghost;
    size_t size;
    
    int rank;
    
    FlowParams<T> flow;


    QVec<T, QVecSize> *Q;

    Force<T> *F;
    //    std::vector<Force<T>> sparseF;

    T *𝜈;

    bool *O;

    ComputeUnit(tNi idi, tNi idj, tNi idk, tNi x, tNi y, tNi z, tNi ghost, FlowParams<T> flow);
    
    ~ComputeUnit();





    inline tNi index(tNi i, tNi j, tNi k);
	Velocity<T> getVelocity(tNi i, tNi j, tNi k);
	Velocity<T> getVelocitySparseF(tNi i, tNi j, tNi k, Force<T> f);
    

    void streaming(Streaming scheme);
	void streaming_simple();
    void streaming_esotwist();

    void collision(Collision scheme);
    void collision_Entropic();
    void collision_EgglesSomers();
    void collision_EgglesSomers_LES();

    void fillForTest();
    void setToZero();


    void checkpoint_read(std::string dirname, std::string unit_name);
    void checkpoint_write(std::string dirname, std::string unit_name);

private:
    std::string get_checkpoint_filename(std::string dirname, std::string unit_name, std::string matrix);
    FILE* fopen_read(std::string dirname, std::string unit_name, std::string matrix);
    FILE* fopen_write(std::string dirname, std::string unit_name, std::string matrix);


};

#include "ComputeUnit.hpp"
