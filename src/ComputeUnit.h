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
    tNi xg0, yg0, zg0;
    tNi xg1, yg1, zg1;
    
    
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

    void fillForTest();
    void setToZero();


    void streaming(Streaming scheme);
	void streaming_simple();
    void streaming_esotwist();

    void collision(Collision scheme);
    void collision_Entropic();
    void collision_EgglesSomers();
    void collision_EgglesSomers_LES();
    
    void moments();

    
    void bounceBackBoundary();

        
        


    void checkpoint_read(std::string dirname, std::string unit_name);
    void checkpoint_write(std::string dirname, std::string unit_name);

private:
    std::string get_checkpoint_filename(std::string dirname, std::string unit_name, std::string matrix);
    FILE* fopen_read(std::string dirname, std::string unit_name, std::string matrix);
    FILE* fopen_write(std::string dirname, std::string unit_name, std::string matrix);

    
    tNi inline dirnQ000(tNi i, tNi j, tNi k);
    tNi inline dirnQ1(tNi i, tNi j, tNi k);
    tNi inline dirnQ2(tNi i, tNi j, tNi k);
    tNi inline dirnQ3(tNi i, tNi j, tNi k);
    tNi inline dirnQ4(tNi i, tNi j, tNi k);
    tNi inline dirnQ5(tNi i, tNi j, tNi k);
    tNi inline dirnQ6(tNi i, tNi j, tNi k);
    tNi inline dirnQ7(tNi i, tNi j, tNi k);
    tNi inline dirnQ8(tNi i, tNi j, tNi k);
    tNi inline dirnQ9(tNi i, tNi j, tNi k);
    tNi inline dirnQ10(tNi i, tNi j, tNi k);
    tNi inline dirnQ11(tNi i, tNi j, tNi k);
    tNi inline dirnQ12(tNi i, tNi j, tNi k);
    tNi inline dirnQ13(tNi i, tNi j, tNi k);
    tNi inline dirnQ14(tNi i, tNi j, tNi k);
    tNi inline dirnQ15(tNi i, tNi j, tNi k);
    tNi inline dirnQ16(tNi i, tNi j, tNi k);
    tNi inline dirnQ17(tNi i, tNi j, tNi k);
    tNi inline dirnQ18(tNi i, tNi j, tNi k);
    tNi inline dirnQ19(tNi i, tNi j, tNi k);
    tNi inline dirnQ20(tNi i, tNi j, tNi k);
    tNi inline dirnQ21(tNi i, tNi j, tNi k);
    tNi inline dirnQ22(tNi i, tNi j, tNi k);
    tNi inline dirnQ23(tNi i, tNi j, tNi k);
    tNi inline dirnQ24(tNi i, tNi j, tNi k);
    tNi inline dirnQ25(tNi i, tNi j, tNi k);
    tNi inline dirnQ26(tNi i, tNi j, tNi k);

};




#include "ComputeUnit.hpp"
#include "Collision.hpp"
#include "Streaming.hpp"
#include "Boundary.hpp"

