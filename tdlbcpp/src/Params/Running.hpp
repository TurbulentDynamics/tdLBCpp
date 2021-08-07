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




struct RunningParams
{
    tStep step = 0;
    double angle = 0;

    tStep num_steps = 1;

    tStep impellerStartupStepsUntilNormalSpeed = 30;

    void update(tStep _step, double _angle){
                
        step = (tStep)_step;
        angle = (double)_angle;

    }
    
    void incrementStep(){
        step ++;
    }
    
    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {

            angle = jsonParams["angle"].asDouble();
            
            step = (Json::UInt64)jsonParams["step"].asUInt64();
            num_steps = (tStep)jsonParams["num_steps"].asUInt64();
            impellerStartupStepsUntilNormalSpeed = (tStep)jsonParams["impellerStartupStepsUntilNormalSpeed"].asUInt64();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
        }
            
    };
    
    
    
    
    
    
    
    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;
            
            jsonParams["name"] = "Running";
            
            jsonParams["step"] = (Json::UInt64)step;
            jsonParams["angle"] = (double)angle;

            jsonParams["num_steps"] = (Json::UInt64)num_steps;
            jsonParams["impellerStartupStepsUntilNormalSpeed"] = (Json::UInt64)impellerStartupStepsUntilNormalSpeed;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
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
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
        }
        
    };
    
    
    
    
    int writeParamsToJsonFile(const std::string filePath) {
        
        
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
    
    
    void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
    
};//end of struct
