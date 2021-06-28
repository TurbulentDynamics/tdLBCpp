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
#include "Params/ComputeUnitParams.hpp"



#include "../../tdLBGeometryRushtonTurbineLib/Sources/tdLBGeometryRushtonTurbineLibCPP/RushtonTurbine.hpp"
#include "../../tdLBGeometryRushtonTurbineLib/Sources/tdLBGeometryRushtonTurbineLibCPP/GeomPolar.hpp"
#include "ComputeUnit.h"
#include "OutputConfig.h"

//TODO: Temporary, different ComputeUnits could have different precision
using useQVecPrecision = float;


//https://stackoverflow.com/questions/16357999/current-date-and-time-as-string
std::string getTimeNowAsString(){

    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, sizeof(buffer), "%Y_%m_%d_%H_%M_%S", timeinfo);
    std::string time_now(buffer);

    //https://www.techiedelight.com/replace-occurrences-character-string-cpp/
    size_t pos;
    while ((pos = time_now.find("-")) != std::string::npos) {time_now.replace(pos, 1, "_");}
    while ((pos = time_now.find(" ")) != std::string::npos) {time_now.replace(pos, 1, "_");}
    while ((pos = time_now.find(":")) != std::string::npos) {time_now.replace(pos, 1, "_");}


    return time_now;
}


std::string formatDirWithTimeAndParams(std::string root, tNi gridX, int re_m, bool les, float uav, tStep step = 0){

    std::string str = root + "_";

    if (step) str += "step_" + std::to_string(step) + "__";

    str += "datetime_" + getTimeNowAsString() + "_";
    str += "gridx_" + std::to_string(gridX) + "_";
    str += "re_" + std::to_string(re_m) + "_";
    str += "les_" + std::to_string(les) + "_";
    str += "uav_" + std::to_string(uav);

    return str;
}



int main(int argc, char* argv[]){

    
    GridParams grid;
    grid.x = 44;
    grid.y = 44;
    grid.z = 44;
    
    tStep num_steps = 20;

    
    grid.ngx = 1;
    grid.ngy = 1;
    grid.ngz = 1;
    
    
    FlowParams<useQVecPrecision> flow;
    flow.initialRho = 8.0;
    flow.reMNonDimensional = 7000.0;
    flow.uav = 0.1;
    flow.g3 = 0.1;
    flow.alpha = 0.97;
    flow.beta = 1.9;

    

    RunningParams running;
    running.angle = 0.0;
    running.step = 0;

    
    Checkpoint checkpoint;
    checkpoint.start_with_checkpoint = 0;
    checkpoint.load_checkpoint_dirname = "checkpoint_step_10000";

    checkpoint.checkpoint_repeat = 10;
    checkpoint.checkpoint_root_dir = "checkpoint_root_dir";

    
    
    int useLES = 0;

 
    //    std::string diskOutputDir = formatDirWithTimeAndParams("run_", grid.x, flow.reMNonDimensional, useLES, flow.uav);

    std::string driveRoot = ".";

    std::string diskOutputDir = "output_debug";

    
    tStep plot_XY_plane_repeat = 10;
    tStep plot_XZ_plane_repeat = 10;
    tStep plot_YZ_plane_repeat = 10;
    
    tStep plot_volume_repeat = 20;
    tStep plot_ml_slices_repeat = 0;

    
    
    
    
    
    
    
    try {
        cxxopts::Options options(argv[0], "Stirred Tank 3d");
        options
        .positional_help("[optional args]")
        .show_positional_help();

        options.add_options()
        ("x,snx", "Number of Cells in x direction ie snx", cxxopts::value<tNi>(grid.x))
        ("y,sny", "Number of Cells in y direction sny=snz", cxxopts::value<tNi>(grid.y))
        //("z,snz", "Number of Cells in z direction sny=snz", cxxopts::value<tNi>(grid.z))

        ("n,num_steps", "Number of Steps", cxxopts::value<tStep>(num_steps))

        ("re_m", "Reynolds Number  (Re_m will be *M_PI/2)", cxxopts::value<useQVecPrecision>(flow.reMNonDimensional))
        ("start_with_checkpoint", "start_with_checkpoint", cxxopts::value<bool>(start_with_checkpoint))
        ("load_checkpoint_dirname", "load_checkpoint_dirname", cxxopts::value<std::string>(load_checkpoint_dirname))
//        ("upscale_factor", "upscale_factor", cxxopts::value<tNi>(upscale_factor))
        ("plot_slice_repeat", "plot_slice_repeat", cxxopts::value<tStep>(plot_XY_plane_repeat))
        ("plot_axis_repeat", "plot_axis_repeat", cxxopts::value<tStep>(plot_XZ_plane_repeat))
        ("plot_full_repeat", "plot_full_repeat", cxxopts::value<tStep>(plot_volume_repeat))
        ("h,help", "Lots of Arguments")
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

    
    
    
    RushtonTurbine rt = RushtonTurbine(int(grid.x));
    
    Extents<tNi> e = Extents<tNi>(0, grid.x, 0, grid.y, 0, grid.z);

    //    RushtonTurbineMidPointCPP<tNi> geom = RushtonTurbineMidPointCPP<tNi>(rt, e);
        RushtonTurbinePolarCPP<tNi, useQVecPrecision> geom = RushtonTurbinePolarCPP<tNi, useQVecPrecision>(rt, e);

    
    geom.generateFixedGeometry();
    geom.generateRotatingGeometry(startingAngle);
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
        
    
    DiskOutputTree outputTree = DiskOutputTree(driveRoot, diskOutputDir, grid, flow.asDouble(), cu);

    
    ComputeUnit<useQVecPrecision, QLen::D3Q19> lb = ComputeUnit<useQVecPrecision, QLen::D3Q19>(cu, flow, outputTree);
    
    
    
    lb.forcing(geomFixed, alpha, beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);
    lb.forcing(geomRotatingNonUpdating, alpha, beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);

    
    double impellerAngle = startingAngle;
    RunningParams runParams;
    for (tStep step=1; step<=num_steps; step++) {

        runParams.update(step, (double)impellerAngle);
        
        lb.collision(EgglesSomers);

        lb.streaming(Simple);


        
        //SetUp OutputFormat
        BinFileFormat binFormat;
        //format.filePath = plotPath;
        binFormat.structName = "tDisk_grid_Q4_V5";
        //format.binFileSizeInStructs set at the end
        binFormat.coordsType = "uint16_t";
        binFormat.hasGridtCoords = 1;
        binFormat.hasColRowtCoords = 0;
        binFormat.QDataType = "float";
        binFormat.QOutputLength = 4;

        
        if (step % plot_XZ_plane_repeat == 0) {
          
            binFormat.cutAt = rt.tankDiameter / 3;
            
            
            lb.template savePlaneXZ<float, 4>(binFormat, runParams);
            
            
            
            //lb.save_XZ_slice<half>(format, grid.y/2, "xz_slice", step);
//            lb.save_XZ_slice<float>(format, grid.y/2, "xz_slice", step);
//            lb.save_XZ_slice<double>(format, grid.y/2, "xz_slice", step);

        }
        

        if (checkpoint_repeat && (step % checkpoint_repeat == 0)) {
            std::string dirname = "checkpoint_test_step_" + std::to_string(step);
//            lb.checkpoint_write(dirname);
        }

        impellerAngle += 0.12;
        geom.updateRotatingGeometry(impellerAngle);
        std::vector<PosPolar<tNi, useQVecPrecision>> geomRotation = geom.returnRotatingGeometry();



        lb.forcing(geomRotating, alpha, beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);

            
        

        
        
        
        
    }
    return 0;

}







