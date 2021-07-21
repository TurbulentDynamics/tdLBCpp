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
        ("g,geom", "Load geometry input json file", cxxopts::value<std::string>(geomJsonPath))
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

    if (inputJsonPath != "") {
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

    } else {

        grid.y = grid.x;
        grid.z = grid.x;

        flow.initialRho = 8.0;
        flow.reMNonDimensional = 7000.0;
        flow.uav = 0.1;

        flow.useLES = 0;

        running.num_steps = 20;

        //        output.add_XY_plane("plot", 10, grid.x/2);
        output.add_XZ_plane("plot_slice", 10, grid.y/3-1);
        output.add_XZ_plane("plot_slice", 10, grid.y/3);
        output.add_XZ_plane("plot_slice", 10, grid.y/3+1);
        //        output.add_YZ_plane("plot", 10, grid.x/2);

        //        output.add_volume("volume", 20);
        //        output.add_XZ_plane("ml_slice", 0, grid.y/3-1);
        //        output.add_XZ_plane("ml_slice", 0, grid.y/3+1);

        checkpoint.start_with_checkpoint = 0;

        checkpoint.checkpoint_repeat = 20;

        //    std::string dir = output.getRunDirWithTimeAndParams("run_", grid.x, flow.reMNonDimensional, flow.useLES, flow.uav);

    }

    if (grid.x < 14 || grid.y < 14 || grid.z < 14) {
        std::cout << "The grid should be larger than 14 cells in every direction" << std::endl;
        exit(1);
    }








    RushtonTurbine rt = RushtonTurbine(int(grid.x));


    Extents<tNi> e = Extents<tNi>(0, grid.x, 0, grid.y, 0, grid.z);

    //    RushtonTurbineMidPointCPP<tNi> geom = RushtonTurbineMidPointCPP<tNi>(rt, e);
    RushtonTurbinePolarCPP<tNi, useQVecPrecision> geom = RushtonTurbinePolarCPP<tNi, useQVecPrecision>(rt, e);


    geom.generateFixedGeometry();
    geom.generateRotatingGeometry(running.angle);
    geom.generateRotatingNonUpdatingGeometry();

    std::vector<PosPolar<tNi, useQVecPrecision>> geomFixed = geom.returnFixedGeometry();
    std::vector<PosPolar<tNi, useQVecPrecision>> geomRotating = geom.returnRotatingGeometry();
    std::vector<PosPolar<tNi, useQVecPrecision>> geomRotatingNonUpdating = geom.returnRotatingNonUpdatingGeometry();



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



    DiskOutputTree outputTree = DiskOutputTree(checkpoint, output);

    FlowParams<double> flowAsDouble = flow.asDouble();
    outputTree.setParams(cu, grid, flowAsDouble, running, output, checkpoint);





    auto lb = ComputeUnit<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple>(cu, flow, outputTree);

    if (checkpointPath != ""){
        lb.checkpoint_read(checkpointPath, "Device");
    }

    lb.forcing(geomFixed, flow.alpha, flow.beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);
    lb.forcing(geomRotatingNonUpdating, flow.alpha, flow.beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);


    int rank = 0;
    Multi_Timer mainTimer(rank);
    mainTimer.set_average_steps(10);
    outputTree.createDir("timer_tmp");


    for (tStep step=running.step; step<=running.num_steps; step++) {

        mainTimer.start_epoch();
        double main_time = mainTimer.time_now();
        double total_time = mainTimer.time_now();


        running.incrementStep();
        running.angle += geom.calcThisStepImpellerIncrement(running.step);


        geom.updateRotatingGeometry(running.angle);
        geomRotating = geom.returnRotatingGeometry();
        main_time = mainTimer.check(0, 0, main_time, "updateRotatingGeometry");

        
        
        lb.collision();
        main_time = mainTimer.check(0, 1, main_time, "Collision");

        lb.streaming();
        main_time = mainTimer.check(0, 2, main_time, "Streaming");

        lb.moments();


        lb.forcing(geomRotating, flow.alpha, flow.beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);
        main_time = mainTimer.check(0, 3, main_time, "Forcing");



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



        for (auto xz: output.XZ_planes){

            if ((running.step == xz.start_at_step) ||
                   (running.step > xz.start_at_step 
                && xz.repeat > 0 
                && (running.step - xz.start_at_step) % xz.repeat == 0)) {

                lb.template savePlaneXZ<float, 4>(xz, binFormat, running);
                main_time = mainTimer.check(0, 4, main_time, "savePlaneXZ");

                lb.calcVorticityXZ(xz.cutAt, running);
            }
        }



        if (checkpoint.checkpoint_repeat && (running.step % checkpoint.checkpoint_repeat == 0)) {

            lb.checkpoint_write("Device", running);
            main_time = mainTimer.check(0, 5, main_time, "Checkpoint");
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







