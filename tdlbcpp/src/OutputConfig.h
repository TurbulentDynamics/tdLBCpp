
//
//  OutputConfig.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Niall OByrnes on 23/07/2018.
//  Copyright © 2018 Niall P. O'Byrnes. All rights reserved.
//

#ifndef Output_Config_hpp
#define Output_Config_hpp

#include <stdlib.h>
#include <vector>


#include "Header.h"


#include "Params/OutputData.hpp"


class OutputConfig {
    
public:
    
    //    OutputConfig(){};
    
    
    void print_output_config_data();
    
    
    
    
    std::vector<Plane> XY_planes;
    void add_XY_plane(std::string dir, tStep repeat, tNi cutAt,
                      int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        Plane p;
        p.name_root = dir;
        p.repeat = repeat;
        p.cutAt = cutAt;
        p.Q_output_len = Q_output_len;
        p.start_at_step = start_at_step;
        p.end_at_repeat = end_at_repeat;
        p.use_half_float = use_half_float;
        p.QDataType = QDataType;
        
        //        plane p = {dir, repeat, cutAt, Q_output_len, start_at_step, end_at_repeat, use_half_float, QDataType};
        XY_planes.push_back(p);
    }
    
    
    
    
    
    
    
    //formally axis
    std::vector<Plane> XZ_planes;
    void add_XZ_plane(const std::string dir, tStep repeat, tNi cutAt,
                      int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        Plane p;
        p.name_root = dir;
        p.repeat = repeat;
        p.cutAt = cutAt;
        p.Q_output_len = Q_output_len;
        p.start_at_step = start_at_step;
        p.end_at_repeat = end_at_repeat;
        p.use_half_float = use_half_float;
        p.QDataType = QDataType;
        
        //        plane p = {dir, repeat, cutAt, Q_output_len, start_at_step, end_at_repeat, use_half_float, QDataType};
        XZ_planes.push_back(p);
    }
    
    
    
    //Formaly slice
    std::vector<Plane> YZ_planes;
    void add_YZ_plane(const std::string dir, tStep repeat, tNi cutAt,
                      int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        Plane p;
        p.name_root = dir;
        p.repeat = repeat;
        p.cutAt = cutAt;
        p.Q_output_len = Q_output_len;
        p.start_at_step = start_at_step;
        p.end_at_repeat = end_at_repeat;
        p.use_half_float = use_half_float;
        p.QDataType = QDataType;
        
        YZ_planes.push_back(p);
    }
    
    
    
    
    
    
    
    std::vector<Angle> capture_at_blade_angle;
    void add_angle(const std::string dir,
                   tStep rotational_capture_repeat, double rotational_capture_behind_impeller_degrees,
                   int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_grid_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_grid_colrow_Q19";
        
        Angle a;
        a.name_root = dir;
        a.repeat = rotational_capture_repeat;
        a.degrees = rotational_capture_behind_impeller_degrees;
        a.Q_output_len = Q_output_len;
        a.start_at_step = start_at_step;
        a.end_at_repeat = end_at_repeat;
        a.use_half_float = use_half_float;
        a.QDataType = QDataType;
        
        capture_at_blade_angle.push_back(a);
        
    }
    
    
    std::vector<PLaneAtAngle> YZ_plane_when_angle;
    void add_YZ_plane_at_angle(const std::string dir, double fixed_axis_capture_behind_impeller_degrees,
                               double tolerance, tNi cutAt, int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        PLaneAtAngle p;
        p.name_root = dir;
        p.degrees = fixed_axis_capture_behind_impeller_degrees;
        p.tolerance = tolerance;
        p.cutAt = cutAt;
        p.Q_output_len = Q_output_len;
        p.start_at_step = start_at_step;
        p.end_at_repeat = end_at_repeat;
        p.use_half_float = use_half_float;
        p.QDataType = QDataType;
        
        YZ_plane_when_angle.push_back(p);
        
    }
    
    
    
    
    
    std::vector<Volume> volumes;
    void add_volume(const std::string dir, tStep plot_full_repeat,
                    int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_grid_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_grid_Q19";
        
        
        Volume v;
        v.name_root = dir;
        v.repeat = plot_full_repeat;
        v.Q_output_len = Q_output_len;
        v.start_at_step = start_at_step;
        v.end_at_repeat = end_at_repeat;
        v.use_half_float = use_half_float;
        v.QDataType = QDataType;
        
        volumes.push_back(v);
        
    }
    
    
    
    std::vector<Sector> sectors;
    //Not Yet Implemenmted
};








#endif














