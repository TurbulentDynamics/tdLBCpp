//
//  main.cpp
//  tdLB
//
//  Created by Niall Ó Broin on 06/10/2020.
//

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <memory>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

//For the overflow error checking. (didnt work with pg compiler)
#include <cfenv>

#include "cxxopts.hpp"

#include "Header.h"
#include "timer.h"
#include "Params/Grid.hpp"
#include "Params/Flow.hpp"
#include "Params/Running.hpp"
#include "Params/Checkpoint.hpp"
#include "Params/OutputParams.hpp"
#include "Params/ComputeUnitParams.hpp"



#include "Sources/tdLBGeometryRushtonTurbineLibCPP/RushtonTurbine.hpp"
#include "Sources/tdLBGeometryRushtonTurbineLibCPP/GeomPolar.hpp"
#include "ComputeUnit.h"

//TODO: Temporary, different ComputeUnits could have different precision
using useQVecPrecision = float;





int main(int argc, char* argv[]){

    std::feclearexcept(FE_OVERFLOW);
    std::feclearexcept(FE_UNDERFLOW);
    std::feclearexcept(FE_DIVBYZERO);



    GridParams grid;
    FlowParams<useQVecPrecision> flow;
    RunningParams running;
    OutputParams output("output_debug");
    CheckpointParams checkpoint;

    std::string inputJsonPath = "";
    std::string geomJsonPath = "";
    std::string checkpointPath = "";
    std::string streaming = "";


    try {
        cxxopts::Options options(argv[0], "Stirred Tank 3d");
        options
        .positional_help("[optional args]")
        .show_positional_help();

        options.add_options()
        ("x,snx", "Number of Cells in x direction ie snx", cxxopts::value<tNi>(grid.x))
        ("j,json", "Load input json file", cxxopts::value<std::string>(inputJsonPath))
        //        ("g,geom", "Load geometry input json file", cxxopts::value<std::string>(geomJsonPath))
        ("c,checkpoint_dir", "Load from Checkpoint directory", cxxopts::value<std::string>(checkpointPath))
        ("s,streaming", "Streaming simple or esoteric", cxxopts::value<std::string>(streaming)->default_value("simple"))
        ("h,help", "Help")
        ;

        options.parse_positional({"input", "output", "positional"});

        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }

    } catch (const cxxopts::OptionException& e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }


    if (checkpointPath != ""){
        inputJsonPath = checkpointPath + "/AllParams.json";
    }

    std::cout << "Debug: inputJsonPath, checkpointPath" << inputJsonPath << checkpointPath << std::endl;

    bool parametersLoadedFromJson = false;
    if (inputJsonPath != "") {
        try {
            std::cout << "Loading " << inputJsonPath << std::endl;

            std::ifstream in(inputJsonPath.c_str());
            Json::Value jsonParams;
            in >> jsonParams;
            in.close();

            grid.getParamsFromJson(jsonParams["GridParams"]);
            flow.getParamsFromJson(jsonParams["FlowParams"]);
            running.getParamsFromJson(jsonParams["RunningParams"]);
            output.getParamsFromJson(jsonParams["OutputParams"]);
            checkpoint.getParamsFromJson(jsonParams["CheckpointParams"]);
            parametersLoadedFromJson = true;
        } catch (std::exception &e) {
            std::cerr << "Exception reached parsing input json: "
            << e.what() << ", will use default parameters" << std::endl;
        }
    }

    if (!parametersLoadedFromJson) {

        grid.x = 120;
        grid.y = grid.x;
        grid.z = grid.x;

        flow.initialRho = 8.0;
        flow.reMNonDimensional = 7300.0;
        flow.uav = 0.1;

        flow.useLES = 0;
        flow.cs0 = 0.12;

        running.num_steps = 100;
        running.impellerStartupStepsUntilNormalSpeed = 20;
    }





    //======== Set up Geometry

    RushtonTurbine rt = RushtonTurbine(int(grid.x));
    flow.calc_nu(rt.impellers[0].blades.outerRadius);
    flow.printParams();


    Extents<tNi> e = Extents<tNi>(0, grid.x, 0, grid.y, 0, grid.z);


    //    RushtonTurbineMidPointCPP<tNi> geom = RushtonTurbineMidPointCPP<tNi>(rt, e);
    RushtonTurbinePolarCPP<tNi, useQVecPrecision> geom = RushtonTurbinePolarCPP<tNi, useQVecPrecision>(rt, e);
    geom.impellerStartupStepsUntilNormalSpeed = running.impellerStartupStepsUntilNormalSpeed;
    useQVecPrecision increment = geom.calcThisStepImpellerIncrement(running.step);
    std::cout << increment<<std::endl;



    // =================== Set Up Output
    if (!parametersLoadedFromJson) {

        output.add_XY_plane("plot_axis", (tStep)1, (tNi)geom.kCenter);

        
        //        output.add_XZ_plane("plot_slice", 10, rt.impellers[0].impellerPosition-1);
        output.add_XZ_plane("plot_slice", 2, rt.impellers[0].impellerPosition);
        //        output.add_XZ_plane("plot_slice", 10, rt.impellers[0].impellerPosition+1);
        //        output.add_YZ_plane("plot", 10, grid.x/2);

        
        //        output.add_volume("volume", 20);
        //        output.add_XZ_plane("ml_slice", 0, grid.y/3-1);
        //        output.add_XZ_plane("ml_slice", 0, grid.y/3+1);

        checkpoint.start_with_checkpoint = 0;

        checkpoint.checkpoint_repeat = 20;

        //    std::string dir = output.getRunDirWithTimeAndParams("run_", grid.x, flow.reMNonDimensional, flow.useLES, flow.uav);

    }



    ComputeUnitParams cu;
    cu.idi = 0;
    cu.idj = 0;
    cu.idk = 0;
    cu.x = grid.x;
    cu.y = grid.y;
    cu.z = grid.z;
    cu.i0 = 0;
    cu.j0 = 0;
    cu.k0 = 0;
    cu.ghost = 1;



    FlowParams<double> flowAsDouble = flow.asDouble();
    DiskOutputTree outputTree = DiskOutputTree(checkpoint, output);
    outputTree.setParams(cu, grid, flowAsDouble, running, output, checkpoint);


    ComputeUnitBase<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL> *lb;
    if (streaming == "simple") {
      lb = new ComputeUnit<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple>(cu, flow, outputTree);
    } else {
      lb = new ComputeUnit<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Esotwist>(cu, flow, outputTree);
    }
    if (checkpointPath != ""){
        lb->checkpoint_read(checkpointPath, "Device");
    } else {
        lb->initialise(flow.initialRho);
    }
    lb->initialiseExcludePoints(geom);






    std::vector<PosPolar<tNi, useQVecPrecision>> wall = geom.getTankWall();
    std::vector<PosPolar<tNi, useQVecPrecision>> baffles = geom.getBaffles(surfaceAndInternal);

    std::vector<PosPolar<tNi, useQVecPrecision>> hub = geom.getImpellerHub(increment, surfaceAndInternal);
    std::vector<PosPolar<tNi, useQVecPrecision>> shaft = geom.getImpellerShaft(increment, surfaceAndInternal);
    std::vector<PosPolar<tNi, useQVecPrecision>> disc = geom.getImpellerDisk(increment, surfaceAndInternal);

    std::vector<PosPolar<tNi, useQVecPrecision>> blades = geom.getImpellerBlades(running.angle, increment, surfaceAndInternal);


    std::vector<PosPolar<tNi, useQVecPrecision>> geomFORCING = wall;
    geomFORCING.insert( geomFORCING.end(), baffles.begin(), baffles.end() );
    geomFORCING.insert( geomFORCING.end(), hub.begin(), hub.end() );
    geomFORCING.insert( geomFORCING.end(), shaft.begin(), shaft.end() );
    geomFORCING.insert( geomFORCING.end(), disc.begin(), disc.end() );
    geomFORCING.insert( geomFORCING.end(), blades.begin(), blades.end() );

    lb->forcing(geomFORCING, flow.alpha, flow.beta);







    int rank = 0;
    Multi_Timer mainTimer(rank);
    mainTimer.set_average_steps(10);
    outputTree.createDir("timer_tmp");


    for (tStep step=running.step; step<=running.num_steps; step++) {

        mainTimer.start_epoch();
        double main_time = mainTimer.time_now();
        double total_time = mainTimer.time_now();




        //=========================================

        running.incrementStep();
        useQVecPrecision increment = geom.calcThisStepImpellerIncrement(running.step);
        running.angle += increment;

        std::cout << "angle " << running.angle << "  increment " << increment << std::endl;



        std::vector<PosPolar<tNi, useQVecPrecision>> geomFORCING = geom.getImpellerBlades(running.angle, increment, surfaceAndInternal);


        if (running.step < geom.impellerStartupStepsUntilNormalSpeed){
            std::vector<PosPolar<tNi, useQVecPrecision>> shaft = geom.getImpellerDisk(increment, surfaceAndInternal);
            std::vector<PosPolar<tNi, useQVecPrecision>> disc = geom.getImpellerDisk(increment, surfaceAndInternal);
            std::vector<PosPolar<tNi, useQVecPrecision>> hub = geom.getImpellerDisk(increment, surfaceAndInternal);
            geomFORCING.insert( geomFORCING.end(), shaft.begin(), shaft.end() );
            geomFORCING.insert( geomFORCING.end(), disc.begin(), disc.end() );
            geomFORCING.insert( geomFORCING.end(), hub.begin(), hub.end() );
        }

        main_time = mainTimer.check(0, 0, main_time, "updateRotatingGeometry");




        //=========================================



        lb->collision();
        main_time = mainTimer.check(0, 1, main_time, "Collision");


        lb->bounceBackBoundary();
        main_time = mainTimer.check(0, 2, main_time, "BounceBack");


        lb->streamingPush();
        main_time = mainTimer.check(0, 3, main_time, "Streaming");

        lb->moments();


        lb->forcing(geomFORCING, flow.alpha, flow.beta);
        main_time = mainTimer.check(0, 4, main_time, "Forcing");



        //SetUp OutputFormat
        BinFileParams binFormat;
        //format.filePath = plotPath; //Set in savePlane* method
        binFormat.structName = "tDisk_grid_Q4_V5";
        //format.binFileSizeInStructs //Set in savePlane* method
        binFormat.coordsType = "uint16_t";
        binFormat.hasGridtCoords = 1;
        binFormat.hasColRowtCoords = 0;
        binFormat.QDataType = "float";
        binFormat.QOutputLength = 4;




        //====================

        for (auto &p: geomFORCING){
            lb->excludeGeomPoints[lb->indexPlusGhost(p.i, p.j, p.k)] = 1;
        }

        for (auto xy: output.XY_planes){
            if (running.step % xy.repeat == 0) {
//                lb->template savePlaneXY<float, 4>(xy, binFormat, running);
//                lb->calcVorticityXY(xy.cutAt, running);
            }
        }



        for (auto xz: output.XZ_planes){
          /*            for (int i = lb->xg - 2; i < lb->xg; i++) {
                for (int k = lb->zg / 2 - 1; k <= lb->zg / 2 + 1; k++) {
                    int j = xz.cutAt;
                    printf("Q[%d,%d,%d] ", i, j, k);
                    for (int l = 0; l < QLen::D3Q19; l++) {
                        printf ("Q%d = %f, ", l+1, lb->Q[lb->index(i, j, k)][l]);
                    }
                    printf("\nF = %f, %f, %f", lb->F[lb->index(i, j, k)].x, lb->F[lb->index(i, j, k)].y, lb->F[lb->index(i, j, k)].z);
                    printf("\n");
                }
                }*/
            if (running.step % xz.repeat == 0) {
//                lb->template savePlaneXZ<float, 4>(xz, binFormat, running);
                lb->calcVorticityXZ(xz.cutAt, running);
            }
        }

        //REMOVE THE ROTATING POINTS.
        for (auto &p: geomFORCING){
            lb->excludeGeomPoints[lb->indexPlusGhost(p.i, p.j, p.k)] = 0;
        }

        //=====================
        //        lb->writeAllOutput(geom, output, binFormat, running);
        main_time = mainTimer.check(0, 5, main_time, "writeAllOutput");




        if (checkpoint.checkpoint_repeat && (running.step % checkpoint.checkpoint_repeat == 0)) {

            lb->checkpoint_write("Device", running);
            main_time = mainTimer.check(0, 6, main_time, "Checkpoint");
        }






        mainTimer.check(1, 0, total_time, "TOTAL STEP");

        //MAINLY FOR FOR TESTING
        mainTimer.print_timer_all_nodes_to_files(step, "timer_tmp");

        tGeomShapeRT revs = running.angle * ((180.0/M_PI)/360);
        printf("Node %2i Step %lu/%lu, Angle: % 1.4E (%.1f revs)    ",  rank, step, running.num_steps, running.angle, revs);
        mainTimer.print_time_left(step, running.num_steps, total_time);

        mainTimer.print_timer(step);




    }//end of main for loop  end of main for loop


    return 0;

}







