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
#include "Params/GridParams.hpp"
#include "Params/FlowParams.hpp"
#include "Params/RunningParams.hpp"
#include "Params/CheckpointParams.hpp"
#include "Params/OutputParams.hpp"
#include "Params/ComputeUnitParams.hpp"



#include "Sources/tdLBGeometryRushtonTurbineLibCPP/RushtonTurbine.hpp"
#include "Sources/tdLBGeometryRushtonTurbineLibCPP/GeomPolar.hpp"
#include "ComputeUnit.h"

// TODO: : Temporary, different ComputeUnits could have different precision
using useQVecPrecision = float;




int main(int argc, char* argv[]){
#ifdef WITH_CPU
    std::cout << "Compiled WITH_CPU defined" << std::endl;
#endif
#ifdef WITH_GPU
    std::cout << "Compiled WITH_GPU defined" << std::endl;
#endif
#ifdef WITH_GPU_MEMSHARED
    std::cout << "Compiled WITH_GPU_MEMSHARED defined" << std::endl;
#endif
    std::feclearexcept(FE_OVERFLOW);
    std::feclearexcept(FE_UNDERFLOW);
    std::feclearexcept(FE_DIVBYZERO);



    GridParams grid;
    FlowParams<useQVecPrecision> flow;
    RunningParams running;
    OutputParams output;
    CheckpointParams checkpoint;
    BinFileParams binFormat;

    std::string inputJsonFile = "";
    std::string checkpointDir = "";
//    std::string geomJsonPath = "";

    std::string streaming = "";


    // MARK: Load json files
    try {
        cxxopts::Options options(argv[0], "Turbulent Dynamics Lattice Boltzmann");
        options
            .positional_help("[optional args]")
            .show_positional_help();

        options.add_options()
        ("i,input_file", "Load input json file", cxxopts::value<std::string>(inputJsonFile))
        ("c,checkpoint_dir", "Load from Checkpoint directory", cxxopts::value<std::string>(checkpointDir))
        //        ("g,geom", "Load geometry input json file", cxxopts::value<std::string>(geomJsonPath))

        ("s,streaming", "Streaming simple or esoteric", cxxopts::value<std::string>(streaming)->default_value("simple"))
        ("x,snx", "Number of Cells in x direction ie snx", cxxopts::value<tNi>(grid.x))

        ("h,help", "Help")
        ;

        options.parse_positional({"input", "output", "positional"});

        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            return(1);
        }


    } catch (const cxxopts::OptionException& e) {

        std::cout << "error parsing options: " << e.what() << std::endl;
        return(1);
    }




    //MARK: Load Checkpoint json and data, or load inputfile and initialise data on ComputeUnit.initialise

    if (checkpointDir != ""){
        inputJsonFile = checkpointDir + "/AllParams.json";
    }


    if (inputJsonFile != "") {

        std::cout << "Loading input from: " << inputJsonFile << std::endl;

        Json::Value jsonParams;

        try {
            std::ifstream in(inputJsonFile.c_str());
            in >> jsonParams;
            in.close();
        } catch (std::exception &e) {
            std::cerr << "Exception reached parsing input json: " << e.what() << std::endl;
            return(1);
        }


        try {
            grid.getParamsFromJson(jsonParams["GridParams"]);
            flow.getParamsFromJson(jsonParams["FlowParams"]);
            running.getParamsFromJson(jsonParams["RunningParams"]);
            checkpoint.getParamsFromJson(jsonParams["CheckpointParams"]);
            binFormat.getParamsFromJson(jsonParams["BinFileParams"]);
            output.getParamsFromJson(jsonParams["OutputParams"]);

        } catch (std::exception &e) {
            std::cerr << "Exception reached parsing input json: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }

    }




    // MARK: Validate Input Params

    int gpuDeviceID = -1;

#if defined(WITH_GPU) || defined(WITH_GPU_MEMSHARED)
<<<<<<< Updated upstream
    unsigned long long size = (grid.x + 2 * (GHOST)) * (grid.y + 2*(GHOST)) * (grid.z + 2*(GHOST));
=======
    unsigned long long size = grid.x * grid.y * grid.z;
    //memRequired = size * precision * (QLen + F + Ob)
>>>>>>> Stashed changes
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
        return(EXIT_WAIVED);
    }
    
    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceBlockingSync));

#endif



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
    cu.ghost = grid.multiStep;
    cu.resolution = 1;


    FlowParams<double> flowAsDouble = flow.asDouble();

    DiskOutputTree outputTree = DiskOutputTree(checkpoint, output);

    outputTree.setParams(cu, grid, flowAsDouble, running, output, checkpoint);



    ComputeUnitBase<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL> *lb;
    if (streaming == "simple") {
        std::cout << "Streaming = Nieve" << std::endl;
#if WITH_GPU == 1
        lb = new ComputeUnit<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Nieve, GPU>(cu, flow, outputTree);
#else
        lb = new ComputeUnit<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Nieve, CPU>(cu, flow, outputTree);
#endif
    } else {
        std::cout << "Streaming = Esoteric" << std::endl;
#if WITH_GPU == 1
        lb = new ComputeUnit<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Esotwist, GPU>(cu, flow, outputTree);
#else
        lb = new ComputeUnit<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Esotwist, CPU>(cu, flow, outputTree);
#endif
    }





    if (checkpointDir == ""){
        lb->initialise(flow.initialRho);
    } else {
        lb->checkpoint_read(checkpointDir, "Device");
    }





    // MARK: Generate Geometry
    // TODO: Need to add geometry load for checkpoint.


    RushtonTurbine rt = RushtonTurbine(int(grid.x));
    flow.calcNuAndRe_m(rt.impellers[0].blades.outerRadius);
    flow.printParams();


    Extents<tNi> e = Extents<tNi>(0, grid.x, 0, grid.y, 0, grid.z);


    //    RushtonTurbineMidPointCPP<tNi> geom = RushtonTurbineMidPointCPP<tNi>(rt, e);
    RushtonTurbinePolarCPP<tNi, useQVecPrecision> geom = RushtonTurbinePolarCPP<tNi, useQVecPrecision>(rt, e);
    geom.impellerStartupStepsUntilNormalSpeed = running.impellerStartupStepsUntilNormalSpeed;
    useQVecPrecision deltaRunningAngle = geom.calcThisStepImpellerIncrement(running.step);


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



    std::string runDir = outputTree.getInitTimeAndParams();
    outputTree.createDir(runDir);

    outputTree.writeAllParamsJson(binFormat, running, "startParams_" + runDir);


    //Set up a directory to store the timing of EVERY node, separate file for each node
    std::string timer_dir_nodes = "timer_" + runDir;

    //Here also write initial params.





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

        lb->setOutputExcludePoints(geomFORCING);

        for (auto xy: output.XY_planes){
            if (xy.repeat && (running.step >= xy.start_at_step) && ((running.step - xy.start_at_step) % xy.repeat == 0)) {
                //                lb.template savePlaneXY<float, 4>(xy, binFormat, running);
                lb->calcVorticityXY(xy.cutAt, running);
            }
        }



        for (auto xz: output.XZ_planes){
            if (xz.repeat && (running.step >= xz.start_at_step) && ((running.step - xz.start_at_step) % xz.repeat == 0)) {
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
        mainTimer.print_timer_all_nodes_to_files(step, timer_dir_nodes);
        //TODO:
        //Add std::out to stdout_runStartTime


        tGeomShapeRT revs = running.angle * ((180.0/M_PI)/360);
        printf("\nNode %2i Step %lu/%lu, Angle: % 1.4E (%.1f revs)    ",  rank, step, running.num_steps, running.angle, revs);
        mainTimer.print_time_left(step, running.num_steps, total_time);

        mainTimer.print_timer(step);




    }//end of main loop  end of main loop





    return 0;

}







