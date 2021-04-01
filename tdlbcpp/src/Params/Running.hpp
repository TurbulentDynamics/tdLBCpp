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
    uint64_t step = 0;

    double angle = 0;

    
    void update(tStep _step, double _angle){
        
        
        step = (uint64_t)_step;
        angle = (double)_angle;
        
        
    
    }
    
    RunningParams getParamFromJson(const std::string filePath){
        
        RunningParams g;
        
        try
        {
            std::ifstream in(filePath.c_str());
            Json::Value jsonParams;
            in >> jsonParams;
            
            g.angle = jsonParams["angle"].asDouble();
            
            g.step = (tStep)jsonParams["step"].asUInt64();
            
            in.close();
            
            
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return g;
        }
        
        return g;
        
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
        
        try {
            
            Json::Value jsonParams;
            
            jsonParams["name"] = "Running";
            
            jsonParams["step"] = (uint64_t)step;
            jsonParams["angle"] = (double)angle;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return "";
        }
    }
    
    void print(){
        
        std::cout
        << " name:" << "RunningParams"
        << " step:" << step
        << " angle:" << angle

        << std::endl;
        
    }
    
};//end of struct
