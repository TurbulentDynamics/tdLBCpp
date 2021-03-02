//
//  define_datastructures.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Niall Ó Broin on 08/01/2019.
//  Copyright © 2019 Nile Ó Broin. All rights reserved.
//

#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "json.h"

#include "BaseParams.h"



template <typename T>
struct FlowParams {

    T initial_rho = 0.0;
    T re_m_nondimensional = 0.0;

    //This is not a flow param but is needed here
    T uav = 0.0;

    T cs0 = 0.0;
    T g3 = 0.8;
    

    T nu = 0.0;
    
    T fx0 = 0.0;
//    Force overallForce = Force<T>(0, 0, 0);
    

    T Re_m = 0.0;
    T Re_f = 0.0;
    T uf = 0.0;


    
    FlowParams get_from_json_filepath(const std::string filepath){

        FlowParams d;

        try
        {
            std::ifstream in(filepath.c_str());
            Json::Value param_json;
            in >> param_json;

            d.initial_rho = (T)param_json["initial_rho"].asDouble();
            d.re_m_nondimensional = (T)param_json["re_m_nondimensional"].asDouble();
            d.uav = (T)param_json["uav"].asDouble();

            d.cs0 = (T)param_json["cs0"].asDouble();

            d.nu = (T)param_json["nu"].asDouble();
            d.g3 = (T)param_json["g3"].asDouble();
            d.fx0 = (T)param_json["fx0"].asDouble();
            d.Re_f = (T)param_json["Re_f"].asDouble();
            d.uf = (T)param_json["uf"].asDouble();
            d.Re_m = (T)param_json["Re_m"].asDouble();
            

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
            Json::Value param_json;

            param_json["name"] = "FlowParams";

            param_json["initial_rho"] = (double)initial_rho;
            param_json["re_m_nondimensional"] = (double)re_m_nondimensional;
            param_json["uav"] = (double)uav;
            param_json["cs0"] = (double)cs0;
            param_json["nu"] = (double)nu;
            param_json["g3"] = (double)g3;
            param_json["fx0"] = (double)fx0;
            param_json["Re_f"] = (double)Re_f;
            param_json["uf"] = (double)uf;
            param_json["Re_m"] = (double)Re_m;

            std::ofstream out(filepath.c_str(), std::ofstream::out);
            out << param_json;
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

        std::cout
        << " name:" << "FlowParams"
        << " initial_rho:" << initial_rho
        << " re_m_nondimensional:" << re_m_nondimensional
        << " uav:" << uav
        << " cs0:" << cs0

        << " nu:" << nu
        << " g3:" << g3
        << " fx0:" << fx0
        << " Re_f:" << Re_f
        << " uf:" << uf
        << " Re_m:" << Re_m
        << std::endl;

    }

    
}; //end of struct
