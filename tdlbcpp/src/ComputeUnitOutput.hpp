//
//  ComputeUnit.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include <iostream>
#include <cerrno>
#include "Tools/toojpeg.h"

#include "ComputeUnit.h"
#include "DiskOutputTree.h"
#include "Params/BinFile.hpp"







template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::initialiseExcludePoints(RushtonTurbinePolarCPP<tNi, T> geom){

    for(auto &p: geom.getExternalPoints()){
        excludeGeomPoints[indexPlusGhost(p.i, p.j, p.k)] = true;
    }
    for(auto &p: geom.getBaffles(surfaceAndInternal)){
        excludeGeomPoints[indexPlusGhost(p.i, p.j, p.k)] = true;
    }


    //increment only metters for {u,v,w}Delta
    float increment = 0.0;

    for(auto &p: geom.getImpellerHub(increment, surfaceAndInternal)){
        excludeGeomPoints[indexPlusGhost(p.i, p.j, p.k)] = true;
    }
    for(auto &p: geom.getImpellerDisk(increment, surfaceAndInternal)){
        excludeGeomPoints[indexPlusGhost(p.i, p.j, p.k)] = true;
    }
    for(auto &p: geom.getImpellerShaft(increment, surfaceAndInternal)){
        excludeGeomPoints[indexPlusGhost(p.i, p.j, p.k)] = true;
    }

};


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
bool ComputeUnitBase<T, QVecSize, MemoryLayout>::hasOutputAtStep(OutputParams output, RunningParams running)
{
    bool hasOutput = false;
    tStep step = running.step;


    for (auto xy: output.XY_planes){

        if ((step == xy.start_at_step) ||
            (step > xy.start_at_step
             && xy.repeat > 0
             && (step - xy.start_at_step) % xy.repeat == 0)) {
            hasOutput = true;
        }
    }


    for (auto xz: output.XZ_planes){

        if ((step == xz.start_at_step) ||
            (step > xz.start_at_step
             && xz.repeat > 0
             && (step - xz.start_at_step) % xz.repeat == 0)) {
            hasOutput = true;

        }
    }

    for (auto yz: output.YZ_planes){

        if ((step == yz.start_at_step) ||
            (step > yz.start_at_step
             && yz.repeat > 0
             && (step - yz.start_at_step) % yz.repeat == 0)) {
            hasOutput = true;
        }
    }

    return hasOutput;
}




static inline std::string formatStep(tStep step){

    std::stringstream sstream;

    sstream << std::setw(8) << std::setfill('0') << patch::to_string(step);

    return sstream.str();
}





template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::calcVorticityXZ(tNi j, RunningParams runParam){


    T *Vort = new T[size];
    bool minInitialized = false, maxInitialized = false;
    T max = 0, min = 0;

    for (tNi i = 1;  i <= xg1; i++) {

        for (tNi k = 1; k <= zg1; k++) {


            if (excludeGeomPoints[index(i,j,k)] == true) continue;


            //              T uxx = T(0.5) * (Q[dirnQ01(i, j, k)].velocity().x - Q[dirnQ02(i, j, k)].velocity().x);
            T uxy = T(0.5) * (Q[dirnQ05(i, j, k)].velocity().x - Q[dirnQ06(i, j, k)].velocity().x);
            T uxz = T(0.5) * (Q[dirnQ03(i, j, k)].velocity().x - Q[dirnQ04(i, j, k)].velocity().x);

            T uyx = T(0.5) * (Q[dirnQ01(i, j, k)].velocity().y - Q[dirnQ02(i, j, k)].velocity().y);
            //              T uyy = T(0.5) * (Q[dirnQ05(i, j, k)].velocity().y - Q[dirnQ06(i, j, k)].velocity().y);
            T uyz = T(0.5) * (Q[dirnQ03(i, j, k)].velocity().y - Q[dirnQ04(i, j, k)].velocity().y);


            T uzx = T(0.5) * (Q[dirnQ01(i, j, k)].velocity().z - Q[dirnQ02(i, j, k)].velocity().z);
            T uzy = T(0.5) * (Q[dirnQ05(i, j, k)].velocity().z - Q[dirnQ06(i, j, k)].velocity().z);
            //              T uzz = T(0.5) * (Q[dirnQ03(i, j, k)].velocity().z - Q[dirnQ04(i, j, k)].velocity().z);


            T uxyuyx = uxy - uyx;
            T uyzuzy = uyz - uzy;
            T uzxuxz = uzx - uxz;

            Vort[index(i,j,k)] = T(log(T(uyzuzy * uyzuzy + uzxuxz * uzxuxz + uxyuyx * uxyuyx)));

            if (!std::isinf(Vort[index(i,j,k)]) && !std::isnan(Vort[index(i,j,k)]) && (!minInitialized || Vort[index(i,j,k)] < min)) {
                min = Vort[index(i,j,k)];
                minInitialized = true;
            }
            if (!std::isinf(Vort[index(i,j,k)]) && !std::isnan(Vort[index(i,j,k)]) && (!maxInitialized || Vort[index(i,j,k)] > max)) {
                max = Vort[index(i,j,k)];
                maxInitialized = true;
            }
        }
    }

    // Saving JPEG
    auto *pict = new unsigned char[xg1 * zg1];


    //Set at min max on step 74, for nx=80, slowstart=200
    //TOFIX DEBUG TODO
//    min = -25.5539;
//    max = -0.681309;



    for (tNi i = 1;  i <= xg1; i++) {
        for (tNi k = 1; k <= zg1; k++) {
            pict[zg1 * (i - 1) + (k - 1)] = floor(255 * ((Vort[index(i,j,k)] - min) / (max - min)));
        }
    }
    std::string plotDir = outputTree.formatXZPlaneDir(runParam.step, j);
    outputTree.createDir(plotDir);
//    std::string jpegPath = outputTree.formatJpegFileNamePath(plotDir);
    std::string jpegPath = "vort.xz." + formatStep(runParam.step) + ".jpeg";


    TooJpeg::openJpeg(jpegPath);
    TooJpeg::writeJpeg(pict, xg1, zg1,
                       false, 90, false, "Debug");
    TooJpeg::closeJpeg();

}





template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::calcVorticityXY(tNi k, RunningParams runParam){


    T *Vort = new T[size];
    bool minInitialized = false, maxInitialized = false;
    T max = 0, min = 0;


    for (tNi i = 1; i <= xg1; i++) {
        for (tNi j = 1;  j <= yg1; j++) {


            if (excludeGeomPoints[index(i,j,k)] == true) continue;


            //              T uxx = T(0.5) * (Q[dirnQ01(i, j, k)].velocity().x - Q[dirnQ02(i, j, k)].velocity().x);
            T uxy = T(0.5) * (Q[dirnQ05(i, j, k)].velocity().x - Q[dirnQ06(i, j, k)].velocity().x);
            T uxz = T(0.5) * (Q[dirnQ03(i, j, k)].velocity().x - Q[dirnQ04(i, j, k)].velocity().x);

            T uyx = T(0.5) * (Q[dirnQ01(i, j, k)].velocity().y - Q[dirnQ02(i, j, k)].velocity().y);
            //              T uyy = T(0.5) * (Q[dirnQ05(i, j, k)].velocity().y - Q[dirnQ06(i, j, k)].velocity().y);
            T uyz = T(0.5) * (Q[dirnQ03(i, j, k)].velocity().y - Q[dirnQ04(i, j, k)].velocity().y);


            T uzx = T(0.5) * (Q[dirnQ01(i, j, k)].velocity().z - Q[dirnQ02(i, j, k)].velocity().z);
            T uzy = T(0.5) * (Q[dirnQ05(i, j, k)].velocity().z - Q[dirnQ06(i, j, k)].velocity().z);
            //              T uzz = T(0.5) * (Q[dirnQ03(i, j, k)].velocity().z - Q[dirnQ04(i, j, k)].velocity().z);



            T uxyuyx = uxy - uyx;
            T uyzuzy = uyz - uzy;
            T uzxuxz = uzx - uxz;

            Vort[index(i,j,k)] = T(log(T(uyzuzy * uyzuzy + uzxuxz * uzxuxz + uxyuyx * uxyuyx)));

            if (!std::isinf(Vort[index(i,j,k)]) && !std::isnan(Vort[index(i,j,k)]) && (!minInitialized || Vort[index(i,j,k)] < min)) {
                min = Vort[index(i,j,k)];
                minInitialized = true;
            }
            if (!std::isinf(Vort[index(i,j,k)]) && !std::isnan(Vort[index(i,j,k)]) && (!maxInitialized || Vort[index(i,j,k)] > max)) {
                max = Vort[index(i,j,k)];
                maxInitialized = true;
            }
        }
    }

    // Saving JPEG
    auto *pict = new unsigned char[xg1 * zg1];

    //Set at min max on step 74, for nx=80, slowstart=200
    //TOFIX DEBUG TODO
    //    min = -25.5539;
    //    max = -0.681309;



    for (tNi i = 1;  i <= xg1; i++) {
        for (tNi j = 1; j <= yg1; j++) {
            pict[yg1 * (j - 1) + (i - 1)] = floor(255 * ((Vort[index(i,j,k)] - min) / (max - min)));
        }
    }
//    std::string plotDir = outputTree.formatXYPlaneDir(runParam.step, k);
//    outputTree.createDir(plotDir);
    //    std::string jpegPath = outputTree.formatJpegFileNamePath(plotDir);
    std::string jpegPath = "vort.xy." + formatStep(runParam.step) + ".jpeg";


    TooJpeg::openJpeg(jpegPath);
    TooJpeg::writeJpeg(pict, xg1, yg1,
                       false, 90, false, "Debug");
    TooJpeg::closeJpeg();

}





template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::writeAllOutput(RushtonTurbinePolarCPP<tNi, T> geom, OutputParams output, BinFileParams binFormat, RunningParams running)
{

    if (!hasOutputAtStep(output, running)) return;


//    for (auto &p: excludeRotating){
//        excludeGeomPoints[index(p.i, p.j, p.k)] = 1;
//    }



//    for (auto xy: output.XY_planes){
//
//        if ((running.step == xy.start_at_step) ||
//            (running.step > xy.start_at_step
//             && xy.repeat > 0
//             && (running.step - xy.start_at_step) % xy.repeat == 0)) {
//
//            lb.template savePlaneXY<float, 4>(xy, binFormat, running);
//        }
//    }


    for (auto xz: output.XZ_planes){

        if ((running.step == xz.start_at_step) ||
            (running.step > xz.start_at_step
             && xz.repeat > 0
             && (running.step - xz.start_at_step) % xz.repeat == 0)) {

            savePlaneXZ<float, 4>(xz, binFormat, running);

            //FOR DEBUGGING
            calcVorticityXZ(xz.cutAt, running);
        }
    }


//    for (auto yz: output.YZ_planes){
//
//        if ((running.step == yz.start_at_step) ||
//            (running.step > yz.start_at_step
//             && yz.repeat > 0
//             && (running.step - yz.start_at_step) % yz.repeat == 0)) {
//
//            lb.template savePlaneYZ<float, 4>(yz, binFormat, running);
//        }
//    }



//
//
//
//    //REMOVE THE ROTATING POINTS.
//    //TODO TOFIX, this might remove points from the hub!!!
//    for (auto &p: excludeRotating){
//        excludeGeomPoints[index(p.i, p.j, p.k)] = 0;
//    }



}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
template <typename tDiskPrecision, int tDiskSize>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::savePlaneXZ(OrthoPlane plane, BinFileParams binFormat, RunningParams runParam){


    tDiskGrid<tDiskPrecision, tDiskSize> *outputBuffer = new tDiskGrid<tDiskPrecision, tDiskSize>[xg * zg];

    tDiskGrid<tDiskPrecision, 3> *F3outputBuffer = new tDiskGrid<tDiskPrecision, 3>[xg*zg];


    long int qVecBufferLen = 0;
    long int F3BufferLen = 0;
    for (tNi i=1; i<=xg1; i++){
        tNi j = plane.cutAt;
        for (tNi k=1; k<=zg1; k++){

            if (excludeGeomPoints[index(i,j,k)] == true) continue;


            tDiskGrid<tDiskPrecision, tDiskSize> tmp;

            //Set position with absolute value
            tmp.iGrid = uint16_t(i0 + i - 1);
            tmp.jGrid = uint16_t(j      - 1);
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


