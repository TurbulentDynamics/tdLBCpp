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


#if WITH_GPU
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#endif


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

// TODO: : Temporary, different ComputeUnits could have different precision
using useQVecPrecision = float;

// FIXME: Should be an input parameter
#define GHOST 1




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

        grid.x = 100;
        grid.y = grid.x;
        grid.z = grid.x;

        flow.initialRho = 8.0;
        flow.reMNonDimensional = 7300.0;
        flow.uav = 0.1;

        flow.useLES = 1;
        flow.cs0 = 0.12;

        running.num_steps = 100;
        running.impellerStartupStepsUntilNormalSpeed = (tStep)(grid.x * 0.2);
    }




// MARK: Validate All Input Params

int gpuDeviceID = -1;

#if defined(WITH_GPU) || defined(WITH_GPU_MEMSHARED)
    unsigned long long size = (grid.x + (GHOST) + grid.y + (GHOST) + grid.z + (GHOST));
    unsigned long long memRequired = size * (sizeof(useQVecPrecision) * (QLen::D3Q19 + 3 + 1) + sizeof(bool) * (1 + 1));

    int numGpus = 0;
    checkCudaErrors(cudaGetDeviceCount(&numGpus));


    for (int i = 0; i < numGpus; i++) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, i));

        printf("GPU Device %d: \"%s\" totalGlobalMem: %d, managedMemory: %d, with compute capability %d.%d\n\n", i, deviceProp.name,  deviceProp.totalGlobalMem, deviceProp.managedMemory, deviceProp.major, deviceProp.minor);

        if (memRequired < deviceProp.totalGlobalMem) {
            #if defined(WITH_GPU_MEMSHARED)
            if (!deviceProp.managedMemory) continue;
            #endif

            gpuDeviceID = i;
        }
    }

    if (gpuDeviceID == -1){
        std::cout << "Cannot find acceptable GPU device, exiting.  Please check log." << std::endl;
        exit(EXIT_WAIVED);
    }

#endif








    // MARK: Set up Geometry

    RushtonTurbine rt = RushtonTurbine(int(grid.x));
    flow.calc_nu(rt.impellers[0].blades.outerRadius);
    flow.printParams();


    Extents<tNi> e = Extents<tNi>(0, grid.x, 0, grid.y, 0, grid.z);


    //    RushtonTurbineMidPointCPP<tNi> geom = RushtonTurbineMidPointCPP<tNi>(rt, e);
    RushtonTurbinePolarCPP<tNi, useQVecPrecision> geom = RushtonTurbinePolarCPP<tNi, useQVecPrecision>(rt, e);
    geom.impellerStartupStepsUntilNormalSpeed = running.impellerStartupStepsUntilNormalSpeed;
    useQVecPrecision deltaRunningAngle = geom.calcThisStepImpellerIncrement(running.step);



    // MARK: Set Up Output and Checkpoint
    if (!parametersLoadedFromJson) {

        output.add_XY_plane("plot_axis", (tStep)5, (tNi)geom.kCenter);

        
        //        output.add_XZ_plane("plot_slice", 10, rt.impellers[0].impellerPosition-1);
        output.add_XZ_plane("plot_slice", (tStep)5, (tNi)rt.impellers[0].impellerPosition);
        //        output.add_XZ_plane("plot_slice", 10, rt.impellers[0].impellerPosition+1);
        //        output.add_YZ_plane("plot", 10, grid.x/2);

        
        //        output.add_volume("volume", 20);
        //        output.add_XZ_plane("ml_slice", 0, grid.y/3 - 1);
        //        output.add_XZ_plane("ml_slice", 0, grid.y/3 + 1);

        checkpoint.start_with_checkpoint = 0;

        checkpoint.checkpoint_repeat = 20;

        //    std::string dir = output.getRunDirWithTimeAndParams("run_", grid.x, flow.reMNonDimensional, flow.useLES, flow.uav);

    }



    // MARK: Initialise ComputeUnit

    ComputeUnitParams cu;
    cu.nodeID = 0;
    cu.deviceID = gpuDeviceID;
    cu.idi = 0;
    cu.idj = 0;
    cu.idk = 0;
    cu.x = grid.x;
    cu.y = grid.y;
    cu.z = grid.z;
    cu.i0 = 0;
    cu.j0 = 0;
    cu.k0 = 0;
    cu.ghost = GHOST;

    FlowParams<double> flowAsDouble = flow.asDouble();
    DiskOutputTree outputTree = DiskOutputTree(checkpoint, output);
    outputTree.setParams(cu, grid, flowAsDouble, running, output, checkpoint);

    ComputeUnitBase<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL> *lb;
    if (streaming == "simple") {
        std::cout << "Streaming = Nieve" << std::endl;
        lb = new ComputeUnit<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple>(cu, flow, outputTree);
    } else {
        std::cout << "Streaming = Esoteric" << std::endl;
        lb = new ComputeUnit<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Esotwist>(cu, flow, outputTree);
    }

    if (checkpointPath != ""){
        lb->checkpoint_read(checkpointPath, "Device");
    } else {
        lb->initialise(flow.initialRho);
    }


    geom.generateFixedGeometry(onSurface);
    std::vector<PosPolar<tNi, useQVecPrecision>> geomFixed = geom.returnFixedGeometry();


    geom.generateRotatingNonUpdatingGeometry(deltaRunningAngle, surfaceAndInternal);
    std::vector<PosPolar<tNi, useQVecPrecision>> geomRotatingNonUpdating = geom.returnRotatingNonUpdatingGeometry();


    geom.generateRotatingGeometry(running.angle, deltaRunningAngle, surfaceAndInternal);
    std::vector<PosPolar<tNi, useQVecPrecision>> geomRotating = geom.returnRotatingGeometry();


    std::vector<PosPolar<tNi, useQVecPrecision>> geomFORCING = geomFixed;
    geomFORCING.insert( geomFORCING.end(), geomRotatingNonUpdating.begin(), geomRotatingNonUpdating.end() );
    geomFORCING.insert( geomFORCING.end(), geomRotating.begin(), geomRotating.end() );

    lb->forcing(geomFORCING, flow.alpha, flow.beta);



    lb->setOutputExcludePoints(geomFixed);
    std::vector<Pos3d<tNi>> externalPoints = geom.getExternalPoints();
    lb->setOutputExcludePoints(externalPoints);




    // MARK: MAIN LOOP

    int rank = 0;
    Multi_Timer mainTimer(rank);
    mainTimer.set_average_steps(10);
    outputTree.createDir("timer_tmp");


    for (tStep step=running.step; step<=running.num_steps; step++) {

        mainTimer.start_epoch();
        double main_time = mainTimer.time_now();
        double total_time = mainTimer.time_now();




        // MARK: GEOMETRY UPDATE

        running.incrementStep();
        useQVecPrecision deltaRunningAngle = geom.calcThisStepImpellerIncrement(running.step);
        running.angle += deltaRunningAngle;

        std::cout << "angle " << running.angle << "  deltaRunningAngle " << deltaRunningAngle << std::endl;



        std::vector<PosPolar<tNi, useQVecPrecision>> geomFORCING = geom.getImpellerBlades(running.angle, deltaRunningAngle, surfaceAndInternal);


        if (running.step < geom.impellerStartupStepsUntilNormalSpeed){

            geom.generateRotatingNonUpdatingGeometry(deltaRunningAngle, surfaceAndInternal);

            std::vector<PosPolar<tNi, useQVecPrecision>> rotatingNonUpdating = geom.returnRotatingNonUpdatingGeometry();

            geomFORCING.insert( geomFORCING.end(), rotatingNonUpdating.begin(), rotatingNonUpdating.end() );


        } else {

            geom.generateRotatingNonUpdatingGeometry(deltaRunningAngle, surfaceAndInternal);

            std::vector<PosPolar<tNi, useQVecPrecision>> rotatingNonUpdating = geom.returnRotatingNonUpdatingGeometry();

            geomFORCING.insert( geomFORCING.end(), rotatingNonUpdating.begin(), rotatingNonUpdating.end() );

        }






        main_time = mainTimer.check(0, 0, main_time, "updateRotatingGeometry");




        // MARK: COLLISION AND STREAMING



        lb->collision();
        main_time = mainTimer.check(0, 1, main_time, "Collision");


        lb->bounceBackBoundary();
        main_time = mainTimer.check(0, 2, main_time, "BounceBack");


        lb->streamingPush();
        main_time = mainTimer.check(0, 3, main_time, "Streaming");

        lb->moments();


        lb->forcing(geomFORCING, flow.alpha, flow.beta);
        main_time = mainTimer.check(0, 4, main_time, "Forcing");






        // MARK: OUTPUT


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




        lb->setOutputExcludePoints(geomFORCING);

        for (auto xy: output.XY_planes){
            if ((running.step == xy.start_at_step) || 
                (running.step > xy.start_at_step) && xy.repeat && ((running.step - xy.start_at_step) % xy.repeat == 0)) {
//                lb.template savePlaneXY<float, 4>(xy, binFormat, running);
                lb->calcVorticityXY(xy.cutAt, running);
            }
        }



        for (auto xz: output.XZ_planes){
            if ((running.step == xz.start_at_step) || 
                (running.step > xz.start_at_step) && xz.repeat && ((running.step - xz.start_at_step) % xz.repeat == 0)) {
//                lb->template savePlaneXZ<float, 4>(xz, binFormat, running);
                lb->calcVorticityXZ(xz.cutAt, running);
            }
        }

        //REMOVE THE ROTATING POINTS.
        lb->unsetOutputExcludePoints(geomFORCING);


        //        lb.writeAllOutput(geom, output, binFormat, running);
        main_time = mainTimer.check(0, 5, main_time, "writeAllOutput");






        // MARK: CHECKPOINT

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




    }//end of main loop  end of main loop





    return 0;

}







