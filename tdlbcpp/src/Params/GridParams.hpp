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

#include <nlohmann/json.hpp>
using json = nlohmann::json;







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

    
    
    
    void getParamsFromJson(const json& jsonParams) {

        
        try
        {

                ngx = (tNi)jsonParams["ngx"].get<uint64_t>();
    ngy = (tNi)jsonParams["ngy"].get<uint64_t>();
    ngz = (tNi)jsonParams["ngz"].get<uint64_t>();
    x = (tNi)jsonParams["x"].get<uint64_t>();
    y = (tNi)jsonParams["y"].get<uint64_t>();
    z = (tNi)jsonParams["z"].get<uint64_t>();
    multiStep = (tNi)jsonParams["multiStep"].get<uint64_t>();
    strMinQVecPrecision = jsonParams["strMinQVecPrecision"].get<std::string>();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in GridParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
    json getJson() const {
        
        try {
            
            json jsonParams;
            
                jsonParams["ngx"] = ngx;
    jsonParams["ngy"] = ngy;
    jsonParams["ngz"] = ngz;
    jsonParams["x"] = x;
    jsonParams["y"] = y;
    jsonParams["z"] = z;
    jsonParams["multiStep"] = multiStep;
    jsonParams["strMinQVecPrecision"] = (std::string)strMinQVecPrecision;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in GridParams: " << e.what() << std::endl;

            return json();
        }
    }
    
    
    
    
    void getParamsFromJsonFile(const std::string filePath) {
        
        try
        {
            std::ifstream in(filePath.c_str());
            json jsonParams;
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
            
            json jsonParams = getJson();
            
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams.dump(4);  // Pretty print with 4 spaces
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
