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


#include "json.h"

#include "Header.h"
#include "GridParams.hpp"
#include "FlowParams.hpp"
#include "../tdLBGeometryRushtonTurbineLib/Sources/tdLBGeometryRushtonTurbineLibCPP/GeomMidPoint.hpp"
#include "ComputeUnit.h"



int main(int argc, const char * argv[]) {

    
    GridParams grid;
    grid.x = 200;
    grid.y = 200;
    grid.z = 200;
    
    grid.ngx = 1;
    grid.ngy = 1;
    grid.ngz = 1;
    
    
    FlowParams<float> flow;
    flow.initial_rho = 8.0;
    flow.re_m_nondimensional = 7000.0;
    flow.uav = 0.1;
    flow.g3 = 0.1;

    
    
    
    let_tStep num_steps = 20;

    bool start_with_checkpoint = 0;
    std::string load_checkpoint_dirname = "checkpoint_step_10000";

    
    let_tStep checkpoint_repeat = 10;
    std::string checkpoint_root_dir = "checkpoint_root_dir";

    std::string output_root_dir = "output_root_dir";
    let_tStep plot_XY_plane_repeat = 10;
    let_tStep plot_XZ_plane_repeat = 10;
    let_tStep plot_YZ_plane_repeat = 10;
    
    let_tStep plot_volume_repeat = 20;
    let_tStep plot_ml_slices_repeat = 0;

    

    
    ComputeUnit<float, QLen::D3Q19> lb = ComputeUnit<float, QLen::D3Q19>(0, 0, 0, grid.x, grid.y, grid.z, 1, flow);


    
    for (tStep step=0; step<num_steps; step++) {

        lb.collision(EgglesSomers);

        lb.streaming(Simple);


        if (step % plot_XZ_plane_repeat) {

            //lb.save_XZ_slice<half>(grid.y/2, "xz_slice", step);
//            lb.save_XZ_slice<float>(grid.y/2, "xz_slice", step);
//            lb.save_XZ_slice<double>(grid.y/2, "xz_slice", step);

        }
        
        
        if (checkpoint_repeat && (step % checkpoint_repeat == 0)) {
            std::string dirname = "checkpoint_test_step_" + std::to_string(step);
//            lb.checkpoint_write(dirname);
        }
    }
    return 0;

}







