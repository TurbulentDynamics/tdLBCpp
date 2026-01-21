//
//  BinFileParams.hpp
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

struct BinFileParams {

        std::string filePath = "NoFilePath";
        std::string name = "NoName";

    //NoNote
        std::string note = "NoNote";
        std::string structName = "tDisk_colrow_Q4";
        tNi binFileSizeInStructs = 0;
        std::string coordsType = "uint16_t";
        bool hasGridtCoords = false;
        bool hasColRowtCoords = true;
        std::string reference = "absolute";
        tNi i0 = 0;
        tNi j0 = 0;
        tNi k0 = 0;
        std::string QDataType = "float";
        int QOutputLength = 4;

    
    
    
        void getParamsFromJson(const nlohmann::json& jsonParams) {

        
        try
        {

                filePath = jsonParams["filePath"].get<std::string>();
        name = jsonParams["name"].get<std::string>();
        note = jsonParams["note"].get<std::string>();
        structName = jsonParams["structName"].get<std::string>();
        binFileSizeInStructs = (tNi)jsonParams["binFileSizeInStructs"].get<uint64_t>();
        coordsType = jsonParams["coordsType"].get<std::string>();
        hasGridtCoords = jsonParams["hasGridtCoords"].get<bool>();
        hasColRowtCoords = jsonParams["hasColRowtCoords"].get<bool>();
        reference = jsonParams["reference"].get<std::string>();
        i0 = (tNi)jsonParams["i0"].get<uint64_t>();
        j0 = (tNi)jsonParams["j0"].get<uint64_t>();
        k0 = (tNi)jsonParams["k0"].get<uint64_t>();
        QDataType = jsonParams["QDataType"].get<std::string>();
        QOutputLength = static_cast<int>(jsonParams["QOutputLength"].get<int>());

            
        }
        catch(const nlohmann::json::exception& e)
        {
            std::cerr << "JSON parsing error in BinFileParams:: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Failed to parse BinFileParams: ") + e.what());
        }
                
    }
    
    
        nlohmann::json getJson() const {
        
        try {
            
            nlohmann::json jsonParams;
            
                jsonParams["filePath"] = static_cast<std::string>(filePath);
        jsonParams["name"] = static_cast<std::string>(name);
        jsonParams["note"] = static_cast<std::string>(note);
        jsonParams["structName"] = static_cast<std::string>(structName);
        jsonParams["binFileSizeInStructs"] = binFileSizeInStructs;
        jsonParams["coordsType"] = static_cast<std::string>(coordsType);
        jsonParams["hasGridtCoords"] = static_cast<bool>(hasGridtCoords);
        jsonParams["hasColRowtCoords"] = static_cast<bool>(hasColRowtCoords);
        jsonParams["reference"] = static_cast<std::string>(reference);
        jsonParams["i0"] = i0;
        jsonParams["j0"] = j0;
        jsonParams["k0"] = k0;
        jsonParams["QDataType"] = static_cast<std::string>(QDataType);
        jsonParams["QOutputLength"] = static_cast<int>(QOutputLength);

            
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
            throw std::runtime_error(std::string("Failed to parse BinFileParams: ") + e.what());
        }
        
    };
    
    
    
    
        int writeParamsToJsonFile(const std::string filePath) {
        
        
        try {
            
            nlohmann::json jsonParams = getJson();
            
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams.dump(4);  // Pretty print with 4 spaces
            out.close();
            
        } catch(const nlohmann::json::exception& e){
            
            std::cerr << "Exception writing json file for BinFileParams: " << e.what() << std::endl;
        }
        
        return 0;
    }
    
    
        void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
