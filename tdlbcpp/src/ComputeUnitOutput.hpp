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
#include "Params/BinFileParams.hpp"



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::setOutputExcludePoints(std::vector<Pos3d<tNi>> geomPoints){

    for (auto &p: geomPoints){
        ExcludeOutputPoints[indexPlusGhost(p.i, p.j, p.k)] = true;
    }
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::setOutputExcludePoints(std::vector<PosPolar<tNi, T>> geomPoints){

    for (auto &p: geomPoints){
        ExcludeOutputPoints[indexPlusGhost(p.i, p.j, p.k)] = true;
    }
}



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::unsetOutputExcludePoints(std::vector<Pos3d<tNi>> geomPoints){

    for (auto &p: geomPoints){
        ExcludeOutputPoints[indexPlusGhost(p.i, p.j, p.k)] = false;
    }
}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::unsetOutputExcludePoints(std::vector<PosPolar<tNi, T>> geomPoints){

    for (auto &p: geomPoints){
        ExcludeOutputPoints[indexPlusGhost(p.i, p.j, p.k)] = false;
    }
}



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




template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, streamingType>::calcVorticityXZ(tNi j, RunningParams runParam){
    using AF = AccessField<T, QVecSize, MemoryLayout, collisionType, streamingType>;

    T *Vort = new T[size];
    bool minInitialized = false, maxInitialized = false;
    T max = 0, min = 0;

    for (tNi i = 1;  i <= xg1; i++) {

        for (tNi k = 1; k <= zg1; k++) {


            if (ExcludeOutputPoints[index(i,j,k)] == true) continue;


            QVec<T, QVecSize> qDirnQ05 = AF::read(*this, i, j, k + 1);
            QVec<T, QVecSize> qDirnQ06 = AF::read(*this, i, j, k - 1);
            T uxy = T(0.5) * (qDirnQ05.velocity().x - qDirnQ06.velocity().x);
            QVec<T, QVecSize> qDirnQ03 = AF::read(*this, i, j + 1, k);
            QVec<T, QVecSize> qDirnQ04 = AF::read(*this, i, j - 1, k);
            T uxz = T(0.5) * (qDirnQ03.velocity().x - qDirnQ04.velocity().x);

            QVec<T, QVecSize> qDirnQ01 = AF::read(*this, i + 1, j, k);
            QVec<T, QVecSize> qDirnQ02 = AF::read(*this, i - 1, j, k);
            T uyx = T(0.5) * (qDirnQ01.velocity().y - qDirnQ02.velocity().y);
            T uyz = T(0.5) * (qDirnQ03.velocity().y - qDirnQ04.velocity().y);


            T uzx = T(0.5) * (qDirnQ01.velocity().z - qDirnQ02.velocity().z);
            T uzy = T(0.5) * (qDirnQ05.velocity().z - qDirnQ06.velocity().z);


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
    //TODO: Fix
    //    min = -25.5539;
    //    max = -0.681309;



    for (tNi i = 1;  i <= xg1; i++) {
        for (tNi k = 1; k <= zg1; k++) {
            pict[zg1 * (i - 1) + (k - 1)] = floor(255 * ((Vort[index(i,j,k)] - min) / (max - min)));
        }
    }

    std::string plotDir = outputTree.formatXZPlaneDir(runParam.step, j);

    std::string jpegPath = plotDir + ".jpeg";


    TooJpeg::openJpeg(jpegPath);
    TooJpeg::writeJpeg(pict, xg1, zg1,
                       false, 90, false, "Debug");
    TooJpeg::closeJpeg();

    delete[] Vort;
    delete[] pict;
}





template<typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, streamingType>::calcVorticityXY(tNi k, RunningParams runParam){
    using AF = AccessField<T, QVecSize, MemoryLayout, collisionType, streamingType>;

    T *Vort = new T[size];
    bool minInitialized = false, maxInitialized = false;
    T max = 0, min = 0;


    for (tNi i = 1; i <= xg1; i++) {
        for (tNi j = 1;  j <= yg1; j++) {


            if (ExcludeOutputPoints[index(i,j,k)] == true) continue;


            QVec<T, QVecSize> qDirnQ05 = AF::read(*this, i, j, k + 1);
            QVec<T, QVecSize> qDirnQ06 = AF::read(*this, i, j, k - 1);
            T uxy = T(0.5) * (qDirnQ05.velocity().x - qDirnQ06.velocity().x);
            QVec<T, QVecSize> qDirnQ03 = AF::read(*this, i, j + 1, k);
            QVec<T, QVecSize> qDirnQ04 = AF::read(*this, i, j - 1, k);
            T uxz = T(0.5) * (qDirnQ03.velocity().x - qDirnQ04.velocity().x);

            QVec<T, QVecSize> qDirnQ01 = AF::read(*this, i + 1, j, k);
            QVec<T, QVecSize> qDirnQ02 = AF::read(*this, i - 1, j, k);
            T uyx = T(0.5) * (qDirnQ01.velocity().y - qDirnQ02.velocity().y);
            T uyz = T(0.5) * (qDirnQ03.velocity().y - qDirnQ04.velocity().y);


            T uzx = T(0.5) * (qDirnQ01.velocity().z - qDirnQ02.velocity().z);
            T uzy = T(0.5) * (qDirnQ05.velocity().z - qDirnQ06.velocity().z);



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
    auto *pict = new unsigned char[xg1 * yg1];


    //Set at min max on step 74, for nx=80, slowstart=200
    //TODO: Fix
    //    min = -25.5539;
    //    max = -0.681309;



    for (tNi i = 1;  i <= xg1; i++) {
        for (tNi j = 1; j <= yg1; j++) {
            pict[(i - 1) + xg1 * (j - 1)] = floor(255 * ((Vort[index(i,j,k)] - min) / (max - min)));
        }
    }

    std::string plotDir = outputTree.formatXYPlaneDir(runParam.step, k);

    std::string jpegPath = plotDir + ".jpeg";


    TooJpeg::openJpeg(jpegPath);
    TooJpeg::writeJpeg(pict, xg1, yg1,
                       false, 90, false, "Debug");
    TooJpeg::closeJpeg();

    delete[] Vort;
    delete[] pict;
}





template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::writeAllOutput(RushtonTurbinePolarCPP<tNi, T> geom, OutputParams output, BinFileParams binFormat, RunningParams running)
{

    if (!hasOutputAtStep(output, running)) return;


    //    for (auto &p: excludeRotating){
    //        ExcludeOutputPoints[index(p.i, p.j, p.k)] = 1;
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
    //    // FIXME: this might remove points from the hub!!!
    //    for (auto &p: excludeRotating){
    //        ExcludeOutputPoints[index(p.i, p.j, p.k)] = 0;
    //    }



}

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
template <typename tDiskPrecision, int tDiskSize>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::savePlaneXZ(OrthoPlaneParams plane, BinFileParams binFormat, RunningParams runParam){


    tDiskGrid<tDiskPrecision, tDiskSize> *outputBuffer = new tDiskGrid<tDiskPrecision, tDiskSize>[xg * zg];

    tDiskGrid<tDiskPrecision, 3> *F3outputBuffer = new tDiskGrid<tDiskPrecision, 3>[xg*zg];


    long int qVecBufferLen = 0;
    long int F3BufferLen = 0;
    for (tNi i=1; i<=xg1; i++){
        tNi j = plane.cutAt;
        for (tNi k=1; k<=zg1; k++){

            if (ExcludeOutputPoints[index(i,j,k)] == true) continue;


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


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void inline ComputeUnitBase<T, QVecSize, MemoryLayout>::saveJpeg(const char *tag, T* Vort, tNi pictX, tNi pictY, tNi border, RunningParams runParam)
{
    bool minInitialized = false, maxInitialized = false;
    T max = 0, min = 0;

    for (tNi i = border;  i < pictX - border; i++) {
        for (tNi j = border; j < pictY - border; j++) {
            T vortValue = Vort[j * pictX + i];
            if (!std::isinf(vortValue) && !std::isnan(vortValue) && (!minInitialized || vortValue < min)) {
                min = vortValue;
                minInitialized = true;
            }
            if (!std::isinf(vortValue) && !std::isnan(vortValue) && (!maxInitialized || vortValue > max)) {
                max = vortValue;
                maxInitialized = true;
            }
        }
    }

    // Saving JPEG
    tNi pictSizeX = pictX - 2*border;
    tNi pictSizeY = pictY - 2*border;
    auto *pict = new unsigned char[pictSizeX * pictSizeY];

    //Set at min max on step 74, for nx=80, slowstart=200
    //TODO: DEBUG
    //    min = -25.5539;
    //    max = -0.681309;



    for (tNi i = 0;  i < pictSizeX; i++) {
        for (tNi j = 0; j < pictSizeY; j++) {
            T vortValue = Vort[(border + j) *  pictX + (border + i)];
            if (std::isinf(vortValue) || std::isnan(vortValue)) {
                pict[pictSizeX * j + i] = 0;
            } else {
                pict[pictSizeX * j + i] = floor(255 * ((vortValue - min) / (max - min)));
            }
        }
    }

    std::string jpegPath = "vort." + std::string(tag) + "." + formatStep(runParam.step) + ".jpeg";


    TooJpeg::openJpeg(jpegPath);
    TooJpeg::writeJpeg(pict, pictSizeX, pictSizeY,
                       false, 90, false, "Debug");
    TooJpeg::closeJpeg();

    delete[] pict;
}
