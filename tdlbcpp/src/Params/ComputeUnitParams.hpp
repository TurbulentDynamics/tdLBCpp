//
//  BinFileParams.hpp
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




struct ComputeUnitParams
{
    
    int idi = 0;
    int idj = 0;
    int idk = 0;
    tNi x = 0;
    tNi y = 0;
    tNi z = 0;
    tNi x0 = 0;
    tNi y0 = 0;
    tNi z0 = 0;
    tNi ghost = 0;
    

        
    ComputeUnitParams getParamFromJson(const std::string filePath) {
        
        ComputeUnitParams p;

        
        try
        {
            std::ifstream in(filePath.c_str());
            Json::Value jsonParams;
            in >> jsonParams;
            
//            mpiRank = jsonParams["mpiRank"].asInt();
            
            p.idi = jsonParams["idi"].asInt();
            p.idj = jsonParams["idj"].asInt();
            p.idk = jsonParams["idk"].asInt();
            
            p.x = jsonParams["x"].asInt();
            p.y = jsonParams["y"].asInt();
            p.z = jsonParams["z"].asInt();
            
            p.x0 = jsonParams["x0"].asInt();
            p.y0 = jsonParams["y0"].asInt();
            p.z0 = jsonParams["z0"].asInt();
    
            p.ghost = jsonParams["ghost"].asInt();
            
            
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
    
    
    
    
    int writeParams(const std::string filePath) {
        
        
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
    
    
    Json::Value getJson() {
        
        
        
        try {
            
            Json::Value jsonParams;

//            jsonParams["note"] = note;

//            jsonParams["mpiRank"] = mpiRank;
            jsonParams["idi"] = idi;
            jsonParams["idj"] = idj;
            jsonParams["idk"] = idk;
            
            jsonParams["x"] = (int)x;
            jsonParams["y"] = (int)y;
            jsonParams["z"] = (int)z;
            
            jsonParams["x0"] = (int)x0;
            jsonParams["y0"] = (int)y0;
            jsonParams["z0"] = (int)z0;
                    
            jsonParams["ghost"] = (int)ghost;
            
            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return "";
        }
    }
    
    
    void printParams() {
        
        std::cout
        << " mpiRank:" << "mpiRank"
        << " idi:" << idi
        << " idj:" << idj
        << " idk:" << idk
        
        
        << " x0:" << x0
        << " y0:" << y0
        << " z0:" << z0
        
        << " x:" << x
        << " y:" << y
        << " z:" << z
        
        << " ghost:" << ghost
        
        << std::endl;
        
    }
    
};//end of struct
