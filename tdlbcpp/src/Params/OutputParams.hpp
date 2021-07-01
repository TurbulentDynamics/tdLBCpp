
//
//  OutputData.hpp.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Niall OByrnes on 23/07/2018.
//  Copyright © 2018 Niall P. O'Byrnes. All rights reserved.
//

#ifndef Output_Data_hpp
#define Output_Data_hpp

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "json.h"




struct OrthoPlane {
    std::string name_root;
    tStep repeat = 0;
    tNi cutAt = 0;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float = 0;
    
    std::string QDataType;

};




struct Volume {
    std::string name_root;
    tStep repeat = 0;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;
};






struct Angle {
    std::string name_root;
    tStep repeat = 0;
    double degrees = 0;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;
};




struct PlaneAtAngle {
    std::string name_root;
    double degrees = 0;
    double tolerance = 0;
    tNi cutAt = 0;
    
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;
};



struct Sector {
    std::string name_root;
    tStep repeat = 0;
    
    
    double angle_infront_blade = 0.0f;
    double angle_behind_blade = 0.0f;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;
};





struct OutputParams {
    
    
    //    OutputConfig(){};
    std::vector<OrthoPlane> XY_planes;
    std::vector<OrthoPlane> XZ_planes;
    std::vector<OrthoPlane> YZ_planes;
    std::vector<Angle> capture_at_blade_angle;
    std::vector<PlaneAtAngle> YZ_plane_when_angle;
    std::vector<Volume> volumes;
    
    
    //Not Yet Implemenmted
    //std::vector<Sector> sectors;
    
    
    
    
    
    void print_output_config_data();
    
    
    void add_XY_plane(std::string dir, tStep repeat, tNi cutAt,
                      int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        OrthoPlane p;
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
    void add_XZ_plane(const std::string dir, tStep repeat, tNi cutAt,
                      int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        OrthoPlane p;
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
    void add_YZ_plane(const std::string dir, tStep repeat, tNi cutAt,
                      int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        OrthoPlane p;
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
    
    
    void add_YZ_plane_at_angle(const std::string dir, double fixed_axis_capture_behind_impeller_degrees,
                               double tolerance, tNi cutAt, int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        PlaneAtAngle p;
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
    
    
    
    


    //TODO
    void getParamFromJson(const std::string filePath){
        
    }


    //TODO
    int writeParams(const std::string filePath){
        return 0;
    }

    
    //TODO
    Json::Value getJson(){
        
        Json::Value jsonParams;

        return jsonParams;
    }

    //TODO
    void print(){
        
    }

        
    
    
};






#endif
