//
//  OrthoPlaneVorticityParams.hpp
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






//

struct OrthoPlaneVorticityParams {

        std::string name_root = "vort";
    int jpegCompression = 100;
    tNi cutAt = 0;
    tStep repeat = 0;
    tStep start_at_step = 0;
    tStep end_at_step = 0;

    
    
    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {

                name_root = (std::string)jsonParams["name_root"].asString();
    jpegCompression = (int)jsonParams["jpegCompression"].asInt();
    cutAt = (tNi)jsonParams["cutAt"].asUInt64();
    repeat = (tStep)jsonParams["repeat"].asUInt64();
    start_at_step = (tStep)jsonParams["start_at_step"].asUInt64();
    end_at_step = (tStep)jsonParams["end_at_step"].asUInt64();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in OrthoPlaneVorticityParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;
            
                jsonParams["name_root"] = (std::string)name_root;
    jsonParams["jpegCompression"] = (int)jpegCompression;
    jsonParams["cutAt"] = (Json::UInt64)cutAt;
    jsonParams["repeat"] = (Json::UInt64)repeat;
    jsonParams["start_at_step"] = (Json::UInt64)start_at_step;
    jsonParams["end_at_step"] = (Json::UInt64)end_at_step;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in OrthoPlaneVorticityParams: " << e.what() << std::endl;

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
            
            std::cerr << "Exception writing json file for OrthoPlaneVorticityParams: " << e.what() << std::endl;
        }
        
        return 0;
    }
    
    
    void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
