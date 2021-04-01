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
    
    
    
    
    GridParams getParamFromJson(const std::string filePath){
        
        GridParams p;
        
        try
        {
            std::ifstream in(filePath.c_str());
            Json::Value jsonParams;
            in >> jsonParams;
            
            p.ngx = (tNi)jsonParams["ngx"].asInt();
            p.ngy = (tNi)jsonParams["ngy"].asInt();
            p.ngz = (tNi)jsonParams["ngz"].asInt();
            
            p.x = (tNi)jsonParams["x"].asInt();
            p.y = (tNi)jsonParams["y"].asInt();
            p.z = (tNi)jsonParams["z"].asInt();
            
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
    
    
    
    
    
    void print(){
        
        std::cout
        << " name:" << "GridParams"
        << " ngx:" << ngx
        << " ngx:" << ngy
        << " ngx:" << ngz
        << " x:" << x
        << " y:" << y
        << " z:" << y
        << std::endl;
        
    }
    
};//end of struct
