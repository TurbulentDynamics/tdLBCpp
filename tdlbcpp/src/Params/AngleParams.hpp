//
//  AngleParams.hpp
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

struct AngleParams {

        std::string name_root = "angle";
    tStep repeat = 0;
    double degrees = 0.0;
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_step = 0;
    bool use_half_float = false;
    std::string QDataType = "float";

    
    
    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {

                name_root = (std::string)jsonParams["name_root"].asString();
    repeat = (tStep)jsonParams["repeat"].asUInt64();
    degrees = (double)jsonParams["degrees"].asDouble();
    Q_output_len = (int)jsonParams["Q_output_len"].asInt();
    start_at_step = (tStep)jsonParams["start_at_step"].asUInt64();
    end_at_step = (tStep)jsonParams["end_at_step"].asUInt64();
    use_half_float = (bool)jsonParams["use_half_float"].asBool();
    QDataType = (std::string)jsonParams["QDataType"].asString();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in AngleParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;
            
                jsonParams["name_root"] = (std::string)name_root;
    jsonParams["repeat"] = (Json::UInt64)repeat;
    jsonParams["degrees"] = (double)degrees;
    jsonParams["Q_output_len"] = (int)Q_output_len;
    jsonParams["start_at_step"] = (Json::UInt64)start_at_step;
    jsonParams["end_at_step"] = (Json::UInt64)end_at_step;
    jsonParams["use_half_float"] = (bool)use_half_float;
    jsonParams["QDataType"] = (std::string)QDataType;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in AngleParams: " << e.what() << std::endl;

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
            
            std::cerr << "Exception writing json file for AngleParams: " << e.what() << std::endl;
        }
        
        return 0;
    }
    
    
    void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
