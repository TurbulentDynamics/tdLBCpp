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

    std::string jsonPath = "";
    std::string geomJsonPath = "";
    std::string checkpointPath = "";
    
    
    try {
        cxxopts::Options options(argv[0], "Stirred Tank 3d");
        options
        .positional_help("[optional args]")
        .show_positional_help();

        options.add_options()
        ("x,snx", "Number of Cells in x direction ie snx", cxxopts::value<tNi>(grid.x))
        ("j,json", "Load input json file", cxxopts::value<std::string>(jsonPath))
        ("g,geom", "Load geometry input json file", cxxopts::value<std::string>(geomJsonPath))
        ("c,checkpoint_dir", "Load from Checkpoint directory", cxxopts::value<std::string>(checkpointPath))
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
        jsonPath = checkpointPath + "/AllParams.json";
    }
        
    /////////////////
    std::cout<<"ADSFASDFASDF"<<jsonPath << checkpointPath<<std::endl;
    
    
    if (jsonPath != "") {
        std::cout << "Loading " << jsonPath << std::endl;
        
        std::ifstream in(jsonPath.c_str());
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
        
        output.add_XY_plane("plot", 10, grid.x/2);
        output.add_XZ_plane("plot_slice", 10, grid.y/3);
        output.add_YZ_plane("plot", 10, grid.x/2);
        
        output.add_volume("volume", 20);
        output.add_XZ_plane("ml_slice", 0, grid.y/3-1);
        output.add_XZ_plane("ml_slice", 0, grid.y/3+1);

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

    
    
    
    
    ComputeUnit<useQVecPrecision, QLen::D3Q19> lb = ComputeUnit<useQVecPrecision, QLen::D3Q19>(cu, flow, outputTree);
    
    if (checkpointPath != ""){
        lb.checkpoint_read(checkpointPath, "Device");
    }
    
    lb.forcing(geomFixed, flow.alpha, flow.beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);
    lb.forcing(geomRotatingNonUpdating, flow.alpha, flow.beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);

    
    
    
    for (tStep step=running.step; step<=running.num_steps; step++) {
        
        running.incrementStep();
        running.angle += geom.calcThisStepImpellerIncrement(running.step);
        
        
        geom.updateRotatingGeometry(running.angle);
        geomRotating = geom.returnRotatingGeometry();

        
        
        
        lb.collision(EgglesSomers);

        lb.streaming(Simple);

        lb.moments();

        
        lb.forcing(geomRotating, flow.alpha, flow.beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);
        
        
        
        
        
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

        
        
        //TODO Cycle though all planes.
        OrthoPlane xzTMP = output.XZ_planes[0];
        if (running.step % xzTMP.repeat == 0) {
            
            lb.template savePlaneXZ<float, 4>(xzTMP, binFormat, running);
            
            
            
            //lb.save_XZ_slice<half>(format, grid.y/2, "xz_slice", step);
//            lb.save_XZ_slice<float>(format, grid.y/2, "xz_slice", step);
//            lb.save_XZ_slice<double>(format, grid.y/2, "xz_slice", step);

        }
        

        
        
        if (checkpoint.checkpoint_repeat && (running.step % checkpoint.checkpoint_repeat == 0)) {
                        
            lb.checkpoint_write("Device", running);
        }

        
        




        
    }
    return 0;

}







