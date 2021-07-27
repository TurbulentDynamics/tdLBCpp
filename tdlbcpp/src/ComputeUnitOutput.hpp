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




template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::calcVorticityXZ(tNi j, RunningParams runParam){


    T *Vort = new T[size];
    bool minInitialized = false, maxInitialized = false;
    T max = 0, min = 0;

    for (tNi i = 1;  i <= xg1; i++) {

        for (tNi k = 1; k <= zg1; k++) {


            if (excludeGeomPoints[index(i,j,k)] == true) continue;


            //              T uxx = T(0.5) * (Q[dirnQ1(i, j, k)].velocity().x - Q[dirnQ2(i, j, k)].velocity().x);
            T uxy = T(0.5) * (Q[dirnQ5(i, j, k)].velocity().x - Q[dirnQ6(i, j, k)].velocity().x);
            T uxz = T(0.5) * (Q[dirnQ3(i, j, k)].velocity().x - Q[dirnQ4(i, j, k)].velocity().x);

            T uyx = T(0.5) * (Q[dirnQ1(i, j, k)].velocity().y - Q[dirnQ2(i, j, k)].velocity().y);
            //              T uyy = T(0.5) * (Q[dirnQ5(i, j, k)].velocity().y - Q[dirnQ6(i, j, k)].velocity().y);
            T uyz = T(0.5) * (Q[dirnQ3(i, j, k)].velocity().y - Q[dirnQ4(i, j, k)].velocity().y);


            T uzx = T(0.5) * (Q[dirnQ1(i, j, k)].velocity().z - Q[dirnQ2(i, j, k)].velocity().z);
            T uzy = T(0.5) * (Q[dirnQ5(i, j, k)].velocity().z - Q[dirnQ6(i, j, k)].velocity().z);
            //              T uzz = T(0.5) * (Q[dirnQ3(i, j, k)].velocity().z - Q[dirnQ4(i, j, k)].velocity().z);


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
    std::cerr << "min: " << min << ", max: " << max << std::endl;

    for (tNi i = 1;  i <= xg1; i++) {
        for (tNi k = 1; k <= zg1; k++) {
            pict[zg1 * (i - 1)+ (k - 1)] = floor(255 * ((Vort[index(i,j,k)] - min) / (max - min)));
        }
    }
    std::string plotDir = outputTree.formatXZPlaneDir(runParam.step, j);
    outputTree.createDir(plotDir);
    std::string jpegPath = outputTree.formatJpegFileNamePath(plotDir);

    TooJpeg::openJpeg(jpegPath);
    TooJpeg::writeJpeg(pict, xg1, zg1,
                       false, 90, false, "Debug");
    TooJpeg::closeJpeg();

}




template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
bool ComputeUnitBase<T, QVecSize, MemoryLayout>::hasOutputAtStep(OutputParams output, tStep step)
{
    bool hasOutput = false;

    for (auto xy: output.XY_planes){

        if ((running.step == xy.start_at_step) ||
            (running.step > xy.start_at_step
             && xy.repeat > 0
             && (running.step - xy.start_at_step) % xy.repeat == 0)) {
            hasOutput = true;
        }
    }


    for (auto xz: output.XZ_planes){

        if ((running.step == xz.start_at_step) ||
            (running.step > xz.start_at_step
             && xz.repeat > 0
             && (running.step - xz.start_at_step) % xz.repeat == 0)) {
            hasOutput = true;

        }
    }

    for (auto yz: output.YZ_planes){

        if ((running.step == yz.start_at_step) ||
            (running.step > yz.start_at_step
             && yz.repeat > 0
             && (running.step - yz.start_at_step) % yz.repeat == 0)) {
            hasOutput = true;
        }
    }

    return true;
}


template <typename T, int QVecSize, MemoryLayoutType MemoryLayout>
void ComputeUnitBase<T, QVecSize, MemoryLayout>::writeAllOutput(RushtonTurbinePolarCPP<tNi, T> geom, OutputParams output, BinFileFormat binFormat, RunningParams running)
{

    if (!output.hasOutputAtStep(output, running.step)) continue;


    std::vector<Pos3d<int>> excludeRotating = geom.getRotatingExcludePoints(running.angle);

    for (auto p: &excludeRotating){
        excludeGeomPoints[index(p.i, p.j, p.k)] = 1;
    }


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

            lb.template savePlaneXZ<float, 4>(xz, binFormat, running);

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






    //REMOVE THE ROTATING POINTS.
    //TODO TOFIX, this might remove points from the hub!!!
    for (auto p: &excludeRotating){
        excludeGeomPoints[index(p.i, p.j, p.k)] = 0;
    }



}


