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

#include <nlohmann/json.hpp>
using json = nlohmann::json;






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
    std::string strQVecPrecision = "notSet";
    std::string strQLength = "notSet";
    std::string strMemoryLayout = "notSet";
    std::string strCollisonAlgo = "notSet";
    std::string strStreamingAlgo = "notSet";
    std::string strCompileFlag = "notSet";

    
    
    ComputeUnitParams() {}

    ComputeUnitParams(int nodeID, int deviceID, int idi, int idj, int idk, tNi x, tNi y, tNi z, tNi i0, tNi j0, tNi k0, tNi ghost)
     : nodeID(nodeID), deviceID(deviceID), idi(idi), idj(idj), idk(idk), x(x), y(y), z(z), i0(i0), j0(j0), k0(k0), ghost(ghost) {}

    
    void getParamsFromJson(const json& jsonParams) {

        
        try
        {

                nodeID = (int)jsonParams["nodeID"].get<int>();
    deviceID = (int)jsonParams["deviceID"].get<int>();
    idi = (int)jsonParams["idi"].get<int>();
    idj = (int)jsonParams["idj"].get<int>();
    idk = (int)jsonParams["idk"].get<int>();
    x = (tNi)jsonParams["x"].get<uint64_t>();
    y = (tNi)jsonParams["y"].get<uint64_t>();
    z = (tNi)jsonParams["z"].get<uint64_t>();
    i0 = (tNi)jsonParams["i0"].get<uint64_t>();
    j0 = (tNi)jsonParams["j0"].get<uint64_t>();
    k0 = (tNi)jsonParams["k0"].get<uint64_t>();
    ghost = (tNi)jsonParams["ghost"].get<uint64_t>();
    resolution = (tNi)jsonParams["resolution"].get<uint64_t>();
    strQVecPrecision = jsonParams["strQVecPrecision"].get<std::string>();
    strQLength = jsonParams["strQLength"].get<std::string>();
    strMemoryLayout = jsonParams["strMemoryLayout"].get<std::string>();
    strCollisonAlgo = jsonParams["strCollisonAlgo"].get<std::string>();
    strStreamingAlgo = jsonParams["strStreamingAlgo"].get<std::string>();
    strCompileFlag = jsonParams["strCompileFlag"].get<std::string>();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in ComputeUnitParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
    json getJson() const {
        
        try {
            
            json jsonParams;
            
                jsonParams["nodeID"] = (int)nodeID;
    jsonParams["deviceID"] = (int)deviceID;
    jsonParams["idi"] = (int)idi;
    jsonParams["idj"] = (int)idj;
    jsonParams["idk"] = (int)idk;
    jsonParams["x"] = x;
    jsonParams["y"] = y;
    jsonParams["z"] = z;
    jsonParams["i0"] = i0;
    jsonParams["j0"] = j0;
    jsonParams["k0"] = k0;
    jsonParams["ghost"] = ghost;
    jsonParams["resolution"] = resolution;
    jsonParams["strQVecPrecision"] = (std::string)strQVecPrecision;
    jsonParams["strQLength"] = (std::string)strQLength;
    jsonParams["strMemoryLayout"] = (std::string)strMemoryLayout;
    jsonParams["strCollisonAlgo"] = (std::string)strCollisonAlgo;
    jsonParams["strStreamingAlgo"] = (std::string)strStreamingAlgo;
    jsonParams["strCompileFlag"] = (std::string)strCompileFlag;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in ComputeUnitParams: " << e.what() << std::endl;

            return json();
        }
    }
    
    
    
    
    void getParamsFromJsonFile(const std::string filePath) {
        
        try
        {
            std::ifstream in(filePath.c_str());
            json jsonParams;
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
            
            json jsonParams = getJson();
            
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams.dump(4);  // Pretty print with 4 spaces
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
