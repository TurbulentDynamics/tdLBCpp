//
//  ComputeUnitParams.hpp
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

struct ComputeUnitParams {

        int nodeID = 0;
    int deviceID = 0;
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
    tNi resolution = 0;
    int gpu_xthreads_per_block = 8;
    int gpu_ythreads_per_block = 8;
    int gpu_zthreads_per_block = 8;
    std::string strQVecPrecision = "notSet";
    std::string strQLength = "notSet";
    std::string strMemoryLayout = "notSet";
    std::string strCollisonAlgo = "notSet";
    std::string strStreamingAlgo = "notSet";
    std::string strCompileFlag = "notSet";

    
    
    ComputeUnitParams() {}

    ComputeUnitParams(int nodeID, int deviceID, int idi, int idj, int idk, tNi x, tNi y, tNi z, tNi i0, tNi j0, tNi k0, tNi ghost)
     : nodeID(nodeID), deviceID(deviceID), idi(idi), idj(idj), idk(idk), x(x), y(y), z(z), i0(i0), j0(j0), k0(k0), ghost(ghost) {}

    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {

                nodeID = (int)jsonParams["nodeID"].asInt();
    deviceID = (int)jsonParams["deviceID"].asInt();
    idi = (int)jsonParams["idi"].asInt();
    idj = (int)jsonParams["idj"].asInt();
    idk = (int)jsonParams["idk"].asInt();
    x = (tNi)jsonParams["x"].asUInt64();
    y = (tNi)jsonParams["y"].asUInt64();
    z = (tNi)jsonParams["z"].asUInt64();
    i0 = (tNi)jsonParams["i0"].asUInt64();
    j0 = (tNi)jsonParams["j0"].asUInt64();
    k0 = (tNi)jsonParams["k0"].asUInt64();
    ghost = (tNi)jsonParams["ghost"].asUInt64();
    resolution = (tNi)jsonParams["resolution"].asUInt64();
    gpu_xthreads_per_block = (int)jsonParams["gpu_xthreads_per_block"].asInt();
    gpu_ythreads_per_block = (int)jsonParams["gpu_ythreads_per_block"].asInt();
    gpu_zthreads_per_block = (int)jsonParams["gpu_zthreads_per_block"].asInt();
    strQVecPrecision = (std::string)jsonParams["strQVecPrecision"].asString();
    strQLength = (std::string)jsonParams["strQLength"].asString();
    strMemoryLayout = (std::string)jsonParams["strMemoryLayout"].asString();
    strCollisonAlgo = (std::string)jsonParams["strCollisonAlgo"].asString();
    strStreamingAlgo = (std::string)jsonParams["strStreamingAlgo"].asString();
    strCompileFlag = (std::string)jsonParams["strCompileFlag"].asString();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in ComputeUnitParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;
            
                jsonParams["nodeID"] = (int)nodeID;
    jsonParams["deviceID"] = (int)deviceID;
    jsonParams["idi"] = (int)idi;
    jsonParams["idj"] = (int)idj;
    jsonParams["idk"] = (int)idk;
    jsonParams["x"] = (Json::UInt64)x;
    jsonParams["y"] = (Json::UInt64)y;
    jsonParams["z"] = (Json::UInt64)z;
    jsonParams["i0"] = (Json::UInt64)i0;
    jsonParams["j0"] = (Json::UInt64)j0;
    jsonParams["k0"] = (Json::UInt64)k0;
    jsonParams["ghost"] = (Json::UInt64)ghost;
    jsonParams["resolution"] = (Json::UInt64)resolution;
    jsonParams["gpu_xthreads_per_block"] = (int)gpu_xthreads_per_block;
    jsonParams["gpu_ythreads_per_block"] = (int)gpu_ythreads_per_block;
    jsonParams["gpu_zthreads_per_block"] = (int)gpu_zthreads_per_block;
    jsonParams["strQVecPrecision"] = (std::string)strQVecPrecision;
    jsonParams["strQLength"] = (std::string)strQLength;
    jsonParams["strMemoryLayout"] = (std::string)strMemoryLayout;
    jsonParams["strCollisonAlgo"] = (std::string)strCollisonAlgo;
    jsonParams["strStreamingAlgo"] = (std::string)strStreamingAlgo;
    jsonParams["strCompileFlag"] = (std::string)strCompileFlag;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in ComputeUnitParams: " << e.what() << std::endl;

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
            
            std::cerr << "Exception writing json file for ComputeUnitParams: " << e.what() << std::endl;
        }
        
        return 0;
    }
    
    
    void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
