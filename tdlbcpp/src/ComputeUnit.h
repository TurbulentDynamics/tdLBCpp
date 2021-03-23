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
#include "FlowParams.hpp"
#include "QVec.hpp"
#include "PlotDirMeta.h"
#include "QVecBinMeta.h"
#include "PlotDir.h"
#include "Output.hpp"
//#include "GlobalStructures.hpp"
#include "../../tdLBGeometryRushtonTurbineLib/Sources/tdLBGeometryRushtonTurbineLibCPP/RushtonTurbine.hpp"
#include "../../tdLBGeometryRushtonTurbineLib/Sources/tdLBGeometryRushtonTurbineLibCPP/GeomPolar.hpp"











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
    
    
    
    
    
    tNi inline index(tNi i, tNi j, tNi k);
    Velocity<T> inline getVelocity(tNi i, tNi j, tNi k);
    Velocity<T> inline getVelocitySparseF(tNi i, tNi j, tNi k, Force<T> f);
    
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
    
    void forcing(std::vector<PosPolar<tNi, T>>, T alfa, T beta, tNi iCenter, tNi kCenter, tNi radius);
    
    
    void bounceBackBoundary();
    void bounceBackBoundaryRight();
    void bounceBackBoundaryLeft();
    void bounceBackBoundaryUp();
    void bounceBackBoundaryDown();
    void bounceBackBoundaryBackward();
    void bounceBackBoundaryForward();
    
    
    
    
    
    
    void checkpoint_read(std::string dirname, std::string unit_name);
    void checkpoint_write(std::string dirname, std::string unit_name);
    
    
    
    template <typename tDiskPrecision, int tDiskSize>
    void savePlaneXZ(OutputDir outDir, int cutAt, tStep step){
        
        
        
        tDiskGrid<tDiskPrecision, tDiskSize> *outputBuffer = new tDiskGrid<tDiskPrecision, tDiskSize>[xg*zg];
        
        
        int bufferLen = 0;
        for (tNi i=1; i<=xg1; i++){
            tNi j = cutAt;
            //            for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){
                
                tDiskGrid<tDiskPrecision, tDiskSize> tmp;
                tmp.iGrid = uint16_t(i - 1);
                tmp.jGrid = uint16_t(j - 1);
                tmp.kGrid = uint16_t(k - 1);
                
#pragma unroll
                for (int l=0; l<tDiskSize; l++){
                    tmp.q[l] = Q[index(i,j,k)].q[l];
                }
                outputBuffer[bufferLen] = tmp;
                bufferLen++;
                
            }
        }
        
        
        std::string plotPath = outDir.get_XY_plane_dir(step, cutAt, tDiskSize);

        outDir.createDir(plotPath);

        
        
        PlotDir p = PlotDir(plotPath, idi, idj, idk);
        std::string qvecPath = p.get_my_Qvec_filename(QvecNames::Qvec);
        
        
        std::cout<< qvecPath<<std::endl;
        

        FILE *fp = fopen(qvecPath.c_str(), "wb");
        fwrite(outputBuffer, sizeof(tDiskGrid<tDiskPrecision, tDiskSize>), bufferLen, fp);
        fclose(fp);
        
        
        delete[] outputBuffer;
        
        
        QVecBinMeta q;
        int has_grid_coords = 1;
        int has_col_row_coords = 0;
        q.set_file_content("tDisk_grid_Q4_V4", bufferLen, "uint16_t", has_grid_coords, has_col_row_coords, "float", tDiskSize);
        q.save_json_to_Qvec_filepath(qvecPath);
        
        
    }
    
    
    
    
    
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
#include "CollisionEgglesSomers.hpp"
#include "CollisionEntropic.hpp"
#include "StreamingNieve.hpp"
#include "StreamingEsoteric.hpp"
#include "Boundary.hpp"
#include "Forcing.hpp"

