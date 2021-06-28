//
//  define_datastructures.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Niall Ó Broin on 08/01/2019.
//  Copyright © 2019 Niall Ó Broin. All rights reserved.
//

#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "json.h"




template <typename T>
struct FlowParams {
    
    T initialRho = 0.0;
    T reMNonDimensional = 0.0;
    
    //This is not a flow param but is needed here
    T uav = 0.0;
    
    T cs0 = 0.0;
    T g3 = 0.8;
    
    
    T nu = 0.0;
    
    T fx0 = 0.0;
    
    
    T Re_m = 0.0;
    T Re_f = 0.0;
    T uf = 0.0;
    
    T alpha = 0.0;
    T beta = 0.0;

    
    
    FlowParams<double> asDouble(){
    
        FlowParams<double> f;
        f.initialRho = (double)uav;
        f.reMNonDimensional = (double)reMNonDimensional;
        
        f.uav = (double)uav;
       
        f.cs0 = (double)cs0;
        f.g3 = (double)g3;
        
        
        f.nu = (double)nu;
        
        f.fx0 = (double)fx0;
        
        
        f.Re_m = (double)Re_m;
        f.Re_f = (double)Re_f;
        f.uf = (double)uf;

        f.alpha = (double)alpha;
        f.beta = (double)beta;
        
        return f;
    }
    
    GridParams getParamFromJson(const std::string filePath){
        
        FlowParams p;
        
        try
        {
            std::ifstream in(filePath.c_str());
            Json::Value jsonParams;
            in >> jsonParams;
            
            p.initialRho = (T)jsonParams["initialRho"].asDouble();
            p.reMNonDimensional = (T)jsonParams["reMNonDimensional"].asDouble();
            p.uav = (T)jsonParams["uav"].asDouble();
            
            p.cs0 = (T)jsonParams["cs0"].asDouble();
            
            p.nu = (T)jsonParams["nu"].asDouble();
            p.g3 = (T)jsonParams["g3"].asDouble();
            p.fx0 = (T)jsonParams["fx0"].asDouble();
            p.Re_f = (T)jsonParams["Re_f"].asDouble();
            p.uf = (T)jsonParams["uf"].asDouble();
            p.Re_m = (T)jsonParams["Re_m"].asDouble();

            p.alpha = (T)jsonParams["alpha"].asDouble();
            p.beta = (T)jsonParams["beta"].asDouble();

            
            in.close();
            
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return p;
        }
        
        return p;
        
    };
    
    
    
    
    int writeParams(const std::string filePath){
        try {
            
            Json::Value jsonParams = getJson();
            
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams;
            out.close();
            
        } catch(std::exception& e){
            
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return 1;
        }
        
        return 0;
    }
    
    
    
    Json::Value getJson(){
        try
        {
            Json::Value jsonParams;
            
            jsonParams["name"] = "FlowParams";
            
            jsonParams["initialRho"] = (double)initialRho;
            jsonParams["reMNonDimensional"] = (double)reMNonDimensional;
            jsonParams["uav"] = (double)uav;
            jsonParams["cs0"] = (double)cs0;
            jsonParams["nu"] = (double)nu;
            jsonParams["g3"] = (double)g3;
            jsonParams["fx0"] = (double)fx0;
            jsonParams["Re_f"] = (double)Re_f;
            jsonParams["uf"] = (double)uf;
            jsonParams["Re_m"] = (double)Re_m;
            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return "";
        }
    }
    
    
    void print(){
        
        std::cout
        << " name:" << "FlowParams"
        << " initialRho:" << initialRho
        << " reMNonDimensional:" << reMNonDimensional
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
