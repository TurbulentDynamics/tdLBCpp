//
//  ComputeUnit.h
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

#if WITH_GPU == 1
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif


#include "Header.h"
#include "Params/Flow.hpp"
#include "Params/ComputeUnitParams.hpp"
#include "Params/BinFile.hpp"
#include "Params/OutputParams.hpp"


#include "QVec.hpp"
#include "DiskOutputTree.h"
#include "Output.hpp"

#include "Sources/tdLBGeometryRushtonTurbineLibCPP/RushtonTurbine.hpp"
#include "Sources/tdLBGeometryRushtonTurbineLibCPP/GeomPolar.hpp"











template <typename T, int QVecSize>
class ComputeUnit {
public:
    
    //Position in the grid
    int idi, idj, idk;
    int mpiRank;

    
    //Size of this ComputeUnit
    tNi x, y, z;
    
    //Starting absolute position in the grid
    tNi i0, j0, k0;

    
    
    tNi xg, yg, zg;
    tNi xg0, yg0, zg0;
    tNi xg1, yg1, zg1;
    
    
    tNi ghost;
    size_t size;
    
    
    FlowParams<T> flow;
    
    
    QVec<T, QVecSize> *Q;

    bool evenStep;
    
    Force<T> *F;
    //    std::vector<Force<T>> sparseF;
        
    T *Nu;
    
    bool *O;
    
    DiskOutputTree outputTree;

    ComputeUnit();
    ComputeUnit(ComputeUnitParams cuJson, FlowParams<T> flow, DiskOutputTree outputTree);
    
    ~ComputeUnit();
    
    
    tNi inline index(tNi i, tNi j, tNi k);
    Velocity<T> inline getVelocity(tNi i, tNi j, tNi k);
    Velocity<T> inline getVelocitySparseF(tNi i, tNi j, tNi k, Force<T> f);

    void setToZero();
    
    
    void streaming(Streaming scheme);
    void streamingNieve();
    void streamingNieve2();
    void streaming_esotwist();
    
    template <Streaming streamingKind>
    void collision(Collision scheme);
    void collision_Entropic();
    template <Streaming streamingKind>
    void collision_EgglesSomers();
    void collision_EgglesSomers_LES();
    
    void moments();
    
    void forcing(std::vector<PosPolar<tNi, T>>, T alfa, T beta, tNi iCenter, tNi kCenter, tNi radius);
    
    
    void bounceBackBoundary();
    void bounceBackBoundaryRight();
    void bounceBackBoundaryLeft();
    void bounceBackBoundaryUp();
    void bounceBackBoundaryDown();
    void bounceBackBoundaryBackward();
    void bounceBackBoundaryForward();
    

    void calcVorticityXZ(tNi j, RunningParams runParam);

    void setGhostSizes();
    void getParamsFromJson(const std::string filePath);
    int writeParamsToJsonFile(const std::string filePath);
    Json::Value getJson();
    void printParams();

        
    
    void checkpoint_read(std::string dirname, std::string unit_name);
    void checkpoint_write(std::string unit_name, RunningParams run);
    
    
    template <typename tDiskPrecision, int tDiskSize>
    void savePlaneXZ(OrthoPlane plane, BinFileParams binFormat, RunningParams runParam){
        
       
        tDiskGrid<tDiskPrecision, tDiskSize> *outputBuffer = new tDiskGrid<tDiskPrecision, tDiskSize>[xg * zg];
        
        tDiskGrid<tDiskPrecision, 3> *F3outputBuffer = new tDiskGrid<tDiskPrecision, 3>[xg*zg];
        
        
        long int qVecBufferLen = 0;
        long int F3BufferLen = 0;
        for (tNi i=1; i<=xg1; i++){
            tNi j = plane.cutAt;
            for (tNi k=1; k<=zg1; k++){
                
                tDiskGrid<tDiskPrecision, tDiskSize> tmp;
                
                //Set position with absolute value
                tmp.iGrid = uint16_t(i0 + i - 1);
                tmp.jGrid = uint16_t(j);
                tmp.kGrid = uint16_t(k0 + k - 1);
                
#pragma unroll
                for (int l=0; l<tDiskSize; l++){
                    tmp.q[l] = Q[index(i,j,k)].q[l];
                }
                outputBuffer[qVecBufferLen] = tmp;
                qVecBufferLen++;
                
                
                if (F[index(i,j,k)].isNotZero()) {
                
                    tDiskGrid<tDiskPrecision, 3> tmp;
                    
                    //Set position with absolute value
                    tmp.iGrid = uint16_t(i0 + i - 1);
                    tmp.jGrid = uint16_t(j);
                    tmp.kGrid = uint16_t(k0 + k - 1);
                    
                    tmp.q[0] = F[index(i,j,k)].x;
                    tmp.q[1] = F[index(i,j,k)].y;
                    tmp.q[2] = F[index(i,j,k)].z;
                    
                    F3outputBuffer[F3BufferLen] = tmp;
                    F3BufferLen++;
                }
                
                
            }
        }
        
        
        std::string plotDir = outputTree.formatXZPlaneDir(runParam.step, plane.cutAt);
        outputTree.createDir(plotDir);


        binFormat.filePath = outputTree.formatQVecBinFileNamePath(plotDir);
        binFormat.binFileSizeInStructs = qVecBufferLen;
        
        
        

        std::cout<< "Writing output to: " << binFormat.filePath <<std::endl;


        outputTree.writeAllParamsJson(binFormat, runParam);


        FILE *fp = fopen(binFormat.filePath.c_str(), "wb");
        fwrite(outputBuffer, sizeof(tDiskGrid<tDiskPrecision, tDiskSize>), qVecBufferLen, fp);
        fclose(fp);

        
        
        //======================
        binFormat.QOutputLength = 3;
        binFormat.binFileSizeInStructs = F3BufferLen;
        binFormat.filePath = outputTree.formatF3BinFileNamePath(plotDir);
        outputTree.writeAllParamsJson(binFormat, runParam);

        
        FILE *fpF3 = fopen(binFormat.filePath.c_str(), "wb");
        fwrite(F3outputBuffer, sizeof(tDiskGrid<tDiskPrecision, 3>), F3BufferLen, fpF3);
        fclose(fpF3);

        
        delete[] outputBuffer;
        delete[] F3outputBuffer;
        
    }
    
    
    
    
    
private:

    FILE* fopen_read(std::string filePath);
    FILE* fopen_write(std::string filePath);

public:    
    
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

template<typename T, int QVecSize, Streaming streaming>
struct AccessField {
    inline static QVec<T, QVecSize> read(ComputeUnit<T, QVecSize> cu, tNi i, tNi j, tNi k);
    inline static void write(ComputeUnit<T, QVecSize> cu, QVec<T, QVecSize> &q, tNi i, tNi j, tNi k);
};

template<typename T, int QVecSize>
struct AccessField<T, QVecSize, Simple> {
    inline static QVec<T, QVecSize> read(ComputeUnit<T, QVecSize> cu, tNi i, tNi j, tNi k) {
        return cu.Q[cu.index(i,j,k)];
    }
    inline static void write(ComputeUnit<T, QVecSize> cu, QVec<T, QVecSize> &q, tNi i, tNi j, tNi k) {
        cu.Q[cu.index(i,j,k)] = q;
    }
};

#include "ComputeUnit.hpp"
#include "CollisionEgglesSomers.hpp"
#include "CollisionEntropic.hpp"
#include "StreamingNieve.hpp"
#include "StreamingEsoteric.hpp"
#include "Boundary.hpp"
#include "Forcing.hpp"

