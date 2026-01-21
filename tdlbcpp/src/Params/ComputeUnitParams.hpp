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

    
        void getParamsFromJson(const nlohmann::json& jsonParams) {

        
        try
        {

                nodeID = static_cast<int>(jsonParams["nodeID"].get<int>());
        deviceID = static_cast<int>(jsonParams["deviceID"].get<int>());
        idi = static_cast<int>(jsonParams["idi"].get<int>());
        idj = static_cast<int>(jsonParams["idj"].get<int>());
        idk = static_cast<int>(jsonParams["idk"].get<int>());
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
        catch(const nlohmann::json::exception& e)
        {
            std::cerr << "JSON parsing error in ComputeUnitParams:: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Failed to parse ComputeUnitParams: ") + e.what());
        }
                
    }
    
    
        nlohmann::json getJson() const {
        
        try {
            
            nlohmann::json jsonParams;
            
                jsonParams["nodeID"] = static_cast<int>(nodeID);
        jsonParams["deviceID"] = static_cast<int>(deviceID);
        jsonParams["idi"] = static_cast<int>(idi);
        jsonParams["idj"] = static_cast<int>(idj);
        jsonParams["idk"] = static_cast<int>(idk);
        jsonParams["x"] = x;
        jsonParams["y"] = y;
        jsonParams["z"] = z;
        jsonParams["i0"] = i0;
        jsonParams["j0"] = j0;
        jsonParams["k0"] = k0;
        jsonParams["ghost"] = ghost;
        jsonParams["resolution"] = resolution;
        jsonParams["strQVecPrecision"] = static_cast<std::string>(strQVecPrecision);
        jsonParams["strQLength"] = static_cast<std::string>(strQLength);
        jsonParams["strMemoryLayout"] = static_cast<std::string>(strMemoryLayout);
        jsonParams["strCollisonAlgo"] = static_cast<std::string>(strCollisonAlgo);
        jsonParams["strStreamingAlgo"] = static_cast<std::string>(strStreamingAlgo);
        jsonParams["strCompileFlag"] = static_cast<std::string>(strCompileFlag);

            
            return jsonParams;
            
        } catch(const nlohmann::json::exception& e) {
            std::cerr << "JSON serialization error: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Failed to serialize params: ") + e.what());
        }
    }
    
    
    
    
        void getParamsFromJsonFile(const std::string filePath) {
        
        try
        {
            std::ifstream in(filePath.c_str());
            nlohmann::json jsonParams;
            in >> jsonParams;
            in.close();
            
            getParamsFromJson(jsonParams);
            
        }
        catch(const nlohmann::json::exception& e)
        {
            std::cerr << "Exception reading from input file: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Failed to parse ComputeUnitParams: ") + e.what());
        }
        
    };
    
    
    
    
        int writeParamsToJsonFile(const std::string filePath) {
        
        
        try {
            
            nlohmann::json jsonParams = getJson();
            
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams.dump(4);  // Pretty print with 4 spaces
            out.close();
            
        } catch(const nlohmann::json::exception& e){
            
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
