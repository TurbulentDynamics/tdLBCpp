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
    tNi i0 = 0;
    tNi j0 = 0;
    tNi k0 = 0;
    tNi ghost = 0;

    ComputeUnitParams() {}

    ComputeUnitParams(int idi, int idj, int idk, tNi x, tNi y, tNi z, tNi i0, tNi j0, tNi k0, tNi ghost)
     : idi(idi), idj(idj), idk(idk), x(x), y(y), z(z), i0(i0), j0(j0), k0(k0), ghost(ghost) {}
        
    void getParamsFromJson(Json::Value jsonParams) {
        
        try
        {
            
//            mpiRank = jsonParams["mpiRank"].asInt();
            
            idi = jsonParams["idi"].asInt();
            idj = jsonParams["idj"].asInt();
            idk = jsonParams["idk"].asInt();
            
            x = jsonParams["x"].asInt();
            y = jsonParams["y"].asInt();
            z = jsonParams["z"].asInt();
            
            i0 = jsonParams["i0"].asInt();
            j0 = jsonParams["j0"].asInt();
            k0 = jsonParams["k0"].asInt();
    
            ghost = jsonParams["ghost"].asInt();
                        
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
        }
        
    };
    


    
    Json::Value getJson() {
        
        try {
            
            Json::Value jsonParams;

//            jsonParams["mpiRank"] = mpiRank;
            jsonParams["idi"] = (int)idi;
            jsonParams["idj"] = (int)idj;
            jsonParams["idk"] = (int)idk;
            
            jsonParams["x"] = (int)x;
            jsonParams["y"] = (int)y;
            jsonParams["z"] = (int)z;
            
            jsonParams["i0"] = (int)i0;
            jsonParams["j0"] = (int)j0;
            jsonParams["k0"] = (int)k0;
                    
            jsonParams["ghost"] = (int)ghost;
            
            
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
