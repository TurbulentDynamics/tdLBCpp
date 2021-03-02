//
//  define_datastructures.h
//  stirred-tank-3d-xcode-cpp
//
//  Created by Nile Ó Broin on 08/01/2019.
//  Copyright © 2019 Nile Ó Broin. All rights reserved.
//

#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "json.h"

#include "Header.h"
#include "BaseParams.h"



template <typename T>
struct PlotDirMeta {
    
    std::string name = "PlotDirMeta";
    std::string func = "";
    std::string dirname = "";
    tNi cut_at = 0;
    int Q_output_length = 0;
    
    
    std::string note = "";
    
    
    int ngx = 0;
    int ngy = 0;
    int ngz = 0;
    
    tNi grid_x = 0;
    tNi grid_y = 0;
    tNi grid_z = 0;
    
    
    tNi file_height = 0;
    tNi file_width = 0;
    tNi total_height = 0;
    tNi total_width = 0;
    
    
    tStep step = 0;
    double teta = 0;
    
    
    T initial_rho = 0;
    T re_m_nondimensional = 0;
    double uav = 0;
    
    
    

    void set_dims(int ngx, int ngy, int ngz, tNi x, tNi y, tNi z){
        
        ngx = ngx;
        ngy = ngy;
        ngz = ngz;
        
        grid_x = x;
        grid_y = y;
        grid_z = z;
    }
    
    
    void set_height_width(tNi file_height, tNi file_width, tNi total_height, tNi total_width){
        file_height = file_height;
        file_width = file_width;
        total_height = total_height;
        total_width = total_width;
    }
    
    
    void set_running(tStep step, double teta){
        teta = teta;
        step = step;
    }
    
    
    void set_flow(T initial_rho, T re_m_nondimensional, T uav){
        initial_rho = initial_rho;
        re_m_nondimensional = re_m_nondimensional;
        uav = uav;
    }
    
    
    void set_note(std::string note){
        note = note;
    }
    
    
    
    void set_plot(std::string func, std::string dirname, int Q_output_length = 4, tNi cut_at = 0){
        func = func;
        dirname = dirname;
        cut_at = cut_at;
        Q_output_length = Q_output_length;
    }
    
    
    
    
    
    
    
    
    PlotDirMeta get_from_json_filepath(const std::string filepath){
        
        PlotDirMeta d;
        
        try
        {
            std::ifstream in(filepath.c_str());
            Json::Value dim_json;
            in >> dim_json;
            
            
            d.func = dim_json["function"].asString();
            d.dirname = dim_json["dirname"].asString();
            d.cut_at = (tNi)dim_json["cut_at"].asInt();
            d.Q_output_length = (int)dim_json["Q_output_length"].asInt();
            d.note = dim_json["note"].asString();
            
            
            d.ngx = (int)dim_json["ngx"].asInt();
            d.ngy = (int)dim_json["ngy"].asInt();
            d.ngz = (int)dim_json["ngz"].asInt();
            
            
            d.grid_x = (tNi)dim_json["grid_x"].asInt();
            d.grid_y = (tNi)dim_json["grid_y"].asInt();
            d.grid_z = (tNi)dim_json["grid_z"].asInt();
            
            
            d.file_height = (tNi)dim_json["file_height"].asInt();
            d.file_width = (tNi)dim_json["file_width"].asInt();
            d.total_height = (tNi)dim_json["total_height"].asInt();
            d.total_width = (tNi)dim_json["total_width"].asInt();
            
            
            d.step = (tStep)dim_json["step"].asInt();
            d.teta = (double)dim_json["teta"].asDouble();
            
            
            d.initial_rho = (T)dim_json["initial_rho"].asDouble();
            d.re_m_nondimensional = (T)dim_json["re_m_nondimensional"].asDouble();
            d.uav = (T)dim_json["uav"].asDouble();
            
            in.close();
            
            return d;
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return d;
        }
    }
    
    
    
    int save_json_to_filepath(const std::string filepath){
        try
        {
            Json::Value dim_json;
            
            dim_json["name"] = name;
            
            dim_json["function"] = func;
            dim_json["dirname"] = dirname;
            dim_json["cut_at"] = (int)cut_at;
            dim_json["Q_output_length"] = (int)Q_output_length;
            
            
            dim_json["note"] = note;
            
            
            dim_json["ngx"] = (int)ngx;
            dim_json["ngy"] = (int)ngy;
            dim_json["ngz"] = (int)ngz;
            
            dim_json["grid_x"] = (int)grid_x;
            dim_json["grid_y"] = (int)grid_y;
            dim_json["grid_z"] = (int)grid_z;
            
            
            dim_json["file_height"] = (int)file_height;
            dim_json["file_width"] = (int)file_width;
            dim_json["total_height"] = (int)total_height;
            dim_json["total_width"] = (int)total_width;
            
            
            dim_json["step"] = (int)step;
            dim_json["teta"] = (double)teta;
            
            
            dim_json["initial_rho"] = (double)initial_rho;
            dim_json["re_m_nondimensional"] = (double)re_m_nondimensional;
            dim_json["uav"] = (double)uav;
            
            
            std::ofstream out(filepath.c_str(), std::ofstream::out);
            out << dim_json;
            out.close();
            
            return 0;
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return 1;
        }
        return 0;
    }
    
    
    
    
    void print(){
        std:: cout
        << " name:" << name
        << " func:" << func
        << " dirname:" << dirname
        << " cut_at:" << cut_at
        << " note:" << note
        << " Q_output_length:" << Q_output_length
        
        
        << " ngx:" << ngx
        << " ngx:" << ngy
        << " ngx:" << ngz
        
        << " grid_x:" << grid_x
        << " grid_y:" << grid_y
        << " grid_z:" << grid_z
        
        << " file_height:" << file_height
        << " file_width:" << file_width
        << " total_height:" << total_height
        << " total_width:" << total_width
        
        << " step:" << step
        << " teta:" << teta
        
        
        << " initial_rho:" << initial_rho
        << " re_m_nondimensional:" << re_m_nondimensional
        << " uav:" << uav
        
        << std::endl;
    }
    
    
    
    
};
    
    
    
    
