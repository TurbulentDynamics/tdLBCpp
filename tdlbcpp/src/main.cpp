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

//    std::string streaming = "";


    // MARK: Load json files
    try {
        cxxopts::Options options(argv[0], "Turbulent Dynamics Lattice Boltzmann");
        options
            .positional_help("[optional args]")
            .show_positional_help();

        options.add_options()
        ("i,input_file", "Load input json file", cxxopts::value<std::string>(inputJsonFile)->default_value("input_debug_gridx60_numSteps20.json"))
        ("c,checkpoint_dir", "Load from Checkpoint directory", cxxopts::value<std::string>(checkpointDir))
        //        ("g,geom", "Load geometry input json file", cxxopts::value<std::string>(geomJsonPath))

        //Debugging
//        ("s,streaming", "Streaming simple or esoteric", cxxopts::value<std::string>(streaming)->default_value("simple"))
//        ("x,snx", "Number of Cells in x direction ie snx", cxxopts::value<tNi>(grid.x))

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
        inputJsonFile = checkpointDir + "/AllParams.0.0.0.device.json";
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
    unsigned long long size = grid.x * grid.y * grid.z;
    //memRequired = size * precision * (QLen + F + Ob)
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

    DiskOutputTree outputTree = DiskOutputTree(output, checkpoint);

    outputTree.setParams(cu, grid, flowAsDouble, running, output, checkpoint);



    ComputeUnitBase<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL> *lb;
    if (flow.streaming == "Simple") {
        std::cout << "Streaming = Simple" << std::endl;
#if WITH_GPU == 1
        lb = new ComputeUnit<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple, GPU>(cu, flow, outputTree);
#else
        lb = new ComputeUnit<useQVecPrecision, QLen::D3Q19, MemoryLayoutIJKL, EgglesSomers, Simple, CPU>(cu, flow, outputTree);
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
    mainTimer.set_average_steps(running.numStepsForAverageCalc);

    std::string runFile = outputTree.getInitTimeAndParams(running.runningDataFilePrefix);
    outputTree.setRunningDataFile(running.runningDataFileDir, runFile);

    outputTree.writeAllParamsJson(binFormat, running, outputTree.runningDataPath);



    for (tStep step=running.step; step<=running.num_steps; step++) {

        mainTimer.start_epoch();
        double main_time = mainTimer.time_now();
        double total_time = mainTimer.time_now();




        // MARK: GEOMETRY UPDATE

        running.incrementStep();
        useQVecPrecision deltaRunningAngle = geom.calcThisStepImpellerIncrement(running.step);
        running.angle += deltaRunningAngle;

        if (step < running.impellerStartupStepsUntilNormalSpeed) {

            std::stringstream sstream;
            sstream << "Angle " << fixed << std::setw(6) << std::setprecision(4) << running.angle << "  deltaRunningAngle " << deltaRunningAngle << std::endl;
            outputTree.writeToRunningDataFileAndPrint(sstream.str());
        }





        std::vector<PosPolar<tNi, useQVecPrecision>> geomFORCING = geomFixed;
        geomFORCING.insert( geomFORCING.end(), geomRotatingNonUpdating.begin(), geomRotatingNonUpdating.end() );
        geom.updateRotatingGeometry(running.angle, deltaRunningAngle, surfaceAndInternal);
        std::vector<PosPolar<tNi, useQVecPrecision>> geomRotating = geom.returnRotatingGeometry();
        geomFORCING.insert( geomFORCING.end(), geomRotating.begin(), geomRotating.end() );






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


        //TODO: This will write BINARY PLOTS
        //                lb.writeAllOutput(geom, output, binFormat, running);
        main_time = mainTimer.check(0, 5, main_time, "writeAllOutput");




        // MARK: CHECKPOINT

        if (checkpoint.checkpointRepeat && (running.step % checkpoint.checkpointRepeat == 0)) {

            lb->checkpoint_write("device", running);
            main_time = mainTimer.check(0, 6, main_time, "Checkpoint");
        }



        mainTimer.check(1, 0, total_time, "TOTAL STEP");



        //Print running updates
        tGeomShapeRT revs = running.angle * ((180.0/M_PI)/360);
        std::stringstream sstream;
        sstream << "Node " << rank;
        sstream << " Step " << step << "/" << running.num_steps;
        sstream << ", Angle: " << std::setprecision(4) << running.angle;
        sstream << " (" << std::setprecision(1) << revs << " revs)    ";

        sstream << mainTimer.timeLeft(step, running.num_steps, total_time) << std::endl;

        outputTree.writeToRunningDataFileAndPrint(sstream.str());

#if WITH_GPU == 1
        size_t mf, ma;
        cudaMemGetInfo(&mf, &ma);
        std::cout << "GPU memory: free: " << mf << " total: " << ma << std::endl;
#endif

        //Print average time per function
        if (step == 1 || (step > 1 && (step % running.numStepsForAverageCalc) == 0)) {
            std::string text = mainTimer.averageAllFunctions(step);
            outputTree.writeToRunningDataFileAndPrint(text);
        }





    }//end of main loop  end of main loop



    return 0;

}







