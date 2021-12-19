//
//  GridParams.hpp
//  Turbulent Dynamics Lattice Boltzmann Cpp
//
//  Created by Niall Ó Broin on 08/01/2019.
//  Copyright © 2019 Niall Ó Broin. All rights reserved.
//

#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "json.h"







    //Grid Params are the maximum extent of the regular lattice.
    

struct GridParams {

        tNi ngx = 1;
    tNi ngy = 1;
    tNi ngz = 1;
    tNi x = 60;
    tNi y = 60;
    tNi z = 60;
    tNi multiStep = 1;
    std::string strMinQVecPrecision = "float";

    
    
    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {

                ngx = (tNi)jsonParams["ngx"].asUInt64();
    ngy = (tNi)jsonParams["ngy"].asUInt64();
    ngz = (tNi)jsonParams["ngz"].asUInt64();
    x = (tNi)jsonParams["x"].asUInt64();
    y = (tNi)jsonParams["y"].asUInt64();
    z = (tNi)jsonParams["z"].asUInt64();
    multiStep = (tNi)jsonParams["multiStep"].asUInt64();
    strMinQVecPrecision = (std::string)jsonParams["strMinQVecPrecision"].asString();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in GridParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;
            
                jsonParams["ngx"] = (Json::UInt64)ngx;
    jsonParams["ngy"] = (Json::UInt64)ngy;
    jsonParams["ngz"] = (Json::UInt64)ngz;
    jsonParams["x"] = (Json::UInt64)x;
    jsonParams["y"] = (Json::UInt64)y;
    jsonParams["z"] = (Json::UInt64)z;
    jsonParams["multiStep"] = (Json::UInt64)multiStep;
    jsonParams["strMinQVecPrecision"] = (std::string)strMinQVecPrecision;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in GridParams: " << e.what() << std::endl;

            return "";
        }
    }
    
    
    
    
    void getParamsFromJsonFile(const std::string filePath) {
        
        try
        {
            std::ifstream in(filePath.c_str());
            Json::Value jsonParams;
            in >> jsonParams;
            in.close();
            
            getParamsFromJson(jsonParams);
            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reading from input file: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
        
    };
    
    
    
    
    int writeParamsToJsonFile(const std::string filePath) {
        
        
        try {
            
            Json::Value jsonParams = getJson();
            
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams;
            out.close();
            
        } catch(std::exception& e){
            
            std::cerr << "Exception writing json file for GridParams: " << e.what() << std::endl;
        }
        
        return 0;
    }
    
    
    void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
