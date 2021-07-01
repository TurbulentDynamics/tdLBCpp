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




//Grid Params are the maximum extent of the regular lattice.
struct GridParams
{
    tNi ngx = 0;
    tNi ngy = 0;
    tNi ngz = 0;
    
    tNi x = 0;
    tNi y = 0;
    tNi z = 0;
    

    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {

            ngx = (tNi)jsonParams["ngx"].asInt();
            ngy = (tNi)jsonParams["ngy"].asInt();
            ngz = (tNi)jsonParams["ngz"].asInt();
            
            x = (tNi)jsonParams["x"].asInt();
            y = (tNi)jsonParams["y"].asInt();
            z = (tNi)jsonParams["z"].asInt();
            
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
        }
                
    }
    
    
    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;
            
            jsonParams["name"] = "GridParams";
            
            jsonParams["ngx"] = (int)ngx;
            jsonParams["ngy"] = (int)ngy;
            jsonParams["ngz"] = (int)ngz;
            
            jsonParams["x"] = (int)x;
            jsonParams["y"] = (int)y;
            jsonParams["z"] = (int)z;
            
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
