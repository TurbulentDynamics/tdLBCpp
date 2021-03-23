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
#include "json.h"

#include "Header.h"
#include "GridParams.hpp"
#include "FlowParams.hpp"
#include "../../tdLBGeometryRushtonTurbineLib/Sources/tdLBGeometryRushtonTurbineLibCPP/RushtonTurbine.hpp"
#include "../../tdLBGeometryRushtonTurbineLib/Sources/tdLBGeometryRushtonTurbineLibCPP/GeomPolar.hpp"
#include "ComputeUnit.h"
#include "OutputConfig.h"
#include "PlotDir.h"


using usePrecision = float;

int main(int argc, char* argv[]){

    
    GridParams grid;
    grid.x = 44;
    grid.y = 44;
    grid.z = 44;
    
    tStep num_steps = 20;

    
    grid.ngx = 1;
    grid.ngy = 1;
    grid.ngz = 1;
    
    
    FlowParams<usePrecision> flow;
    flow.initial_rho = 8.0;
    flow.re_m_nondimensional = 7000.0;
    flow.uav = 0.1;
    flow.g3 = 0.1;

    



    bool start_with_checkpoint = 0;
    std::string load_checkpoint_dirname = "checkpoint_step_10000";

    
    tStep checkpoint_repeat = 10;
    std::string checkpoint_root_dir = "checkpoint_root_dir";

    std::string output_root_dir = "output_root_dir";
    tStep plot_XY_plane_repeat = 10;
    tStep plot_XZ_plane_repeat = 10;
    tStep plot_YZ_plane_repeat = 10;
    
    tStep plot_volume_repeat = 20;
    tStep plot_ml_slices_repeat = 0;

    

    
    double startingAngle = 0.0;
    usePrecision alpha = 0.97;
    usePrecision beta = 1.9;
    
    
    
    
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

        ("re_m", "Reynolds Number  (Re_m will be *M_PI/2)", cxxopts::value<usePrecision>(flow.re_m_nondimensional))
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

    
    
    OutputDir outDir = OutputDir(output_root_dir, grid);
    
    
    
    RushtonTurbine rt = RushtonTurbine(int(grid.x));
    
    Extents<tNi> e = Extents<tNi>(0, grid.x, 0, grid.y, 0, grid.z);

//    RushtonTurbineMidPointCPP<tNi> geom = RushtonTurbineMidPointCPP<tNi>(rt, e);
    RushtonTurbinePolarCPP<tNi, usePrecision> geom = RushtonTurbinePolarCPP<tNi, usePrecision>(rt, e);

    
    geom.generateFixedGeometry();
    geom.generateRotatingGeometry(startingAngle);
    geom.generateRotatingNonUpdatingGeometry();

    std::vector<PosPolar<tNi, usePrecision>> geomFixed = geom.returnFixedGeometry();
    std::vector<PosPolar<tNi, usePrecision>> geomRotating = geom.returnRotatingGeometry();
    std::vector<PosPolar<tNi, usePrecision>> geomRotatingNonUpdating = geom.returnRotatingNonUpdatingGeometry();

    
        
    ComputeUnit<usePrecision, QLen::D3Q19> lb = ComputeUnit<usePrecision, QLen::D3Q19>(5, 5, 5, grid.x, grid.y, grid.z, 1, flow);

    
    lb.forcing(geomFixed, alpha, beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);
    lb.forcing(geomRotatingNonUpdating, alpha, beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);

    
    double impellerAngle = startingAngle;
    for (tStep step=1; step<=num_steps; step++) {

        lb.collision(EgglesSomers);

        lb.streaming(Simple);


        if (step % plot_XZ_plane_repeat) {
            
            int cutAt = rt.tankDiameter / 3;
            
            lb.template savePlaneXZ<float, 4>(outDir, cutAt, step);
            
            
            
            //lb.save_XZ_slice<half>(grid.y/2, "xz_slice", step);
//            lb.save_XZ_slice<float>(grid.y/2, "xz_slice", step);
//            lb.save_XZ_slice<double>(grid.y/2, "xz_slice", step);

        }
        
        
        if (checkpoint_repeat && (step % checkpoint_repeat == 0)) {
            std::string dirname = "checkpoint_test_step_" + std::to_string(step);
//            lb.checkpoint_write(dirname);
        }

        impellerAngle += 0.12;
        geom.updateRotatingGeometry(impellerAngle);
        std::vector<PosPolar<tNi, usePrecision>> geomRotation = geom.returnRotatingGeometry();
        
        

        lb.forcing(geomRotating, alpha, beta, geom.iCenter, geom.kCenter, geom.turbine.tankDiameter/2);

            
        

        
        
        
        
    }
    return 0;

}







