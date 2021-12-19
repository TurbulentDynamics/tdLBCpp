//
//  PlaneAtAngle.hpp
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

struct PlaneAtAngle {

        std::string name_root = "plane_at_angle";
    double degrees = 0.0;
    double tolerance = 0.0;
    tNi cutAt = 0;
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float = false;
    std::string QDataType = "float";

    
    
    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {

                name_root = (std::string)jsonParams["name_root"].asString();
    degrees = (double)jsonParams["degrees"].asDouble();
    tolerance = (double)jsonParams["tolerance"].asDouble();
    cutAt = (tNi)jsonParams["cutAt"].asUInt64();
    Q_output_len = (int)jsonParams["Q_output_len"].asInt();
    start_at_step = (tStep)jsonParams["start_at_step"].asUInt64();
    end_at_repeat = (tStep)jsonParams["end_at_repeat"].asUInt64();
    use_half_float = (bool)jsonParams["use_half_float"].asBool();
    QDataType = (std::string)jsonParams["QDataType"].asString();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in PlaneAtAngle: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;
            
                jsonParams["name_root"] = (std::string)name_root;
    jsonParams["degrees"] = (double)degrees;
    jsonParams["tolerance"] = (double)tolerance;
    jsonParams["cutAt"] = (Json::UInt64)cutAt;
    jsonParams["Q_output_len"] = (int)Q_output_len;
    jsonParams["start_at_step"] = (Json::UInt64)start_at_step;
    jsonParams["end_at_repeat"] = (Json::UInt64)end_at_repeat;
    jsonParams["use_half_float"] = (bool)use_half_float;
    jsonParams["QDataType"] = (std::string)QDataType;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in PlaneAtAngle: " << e.what() << std::endl;

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
            
            std::cerr << "Exception writing json file for PlaneAtAngle: " << e.what() << std::endl;
        }
        
        return 0;
    }
    
    
    void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
