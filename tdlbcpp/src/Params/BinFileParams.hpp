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

#include "json.h"






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

    
    
    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {

                filePath = (std::string)jsonParams["filePath"].asString();
    name = (std::string)jsonParams["name"].asString();
    note = (std::string)jsonParams["note"].asString();
    structName = (std::string)jsonParams["structName"].asString();
    binFileSizeInStructs = (tNi)jsonParams["binFileSizeInStructs"].asUInt64();
    coordsType = (std::string)jsonParams["coordsType"].asString();
    hasGridtCoords = (bool)jsonParams["hasGridtCoords"].asBool();
    hasColRowtCoords = (bool)jsonParams["hasColRowtCoords"].asBool();
    reference = (std::string)jsonParams["reference"].asString();
    i0 = (tNi)jsonParams["i0"].asUInt64();
    j0 = (tNi)jsonParams["j0"].asUInt64();
    k0 = (tNi)jsonParams["k0"].asUInt64();
    QDataType = (std::string)jsonParams["QDataType"].asString();
    QOutputLength = (int)jsonParams["QOutputLength"].asInt();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in BinFileParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;
            
                jsonParams["filePath"] = (std::string)filePath;
    jsonParams["name"] = (std::string)name;
    jsonParams["note"] = (std::string)note;
    jsonParams["structName"] = (std::string)structName;
    jsonParams["binFileSizeInStructs"] = (Json::UInt64)binFileSizeInStructs;
    jsonParams["coordsType"] = (std::string)coordsType;
    jsonParams["hasGridtCoords"] = (bool)hasGridtCoords;
    jsonParams["hasColRowtCoords"] = (bool)hasColRowtCoords;
    jsonParams["reference"] = (std::string)reference;
    jsonParams["i0"] = (Json::UInt64)i0;
    jsonParams["j0"] = (Json::UInt64)j0;
    jsonParams["k0"] = (Json::UInt64)k0;
    jsonParams["QDataType"] = (std::string)QDataType;
    jsonParams["QOutputLength"] = (int)QOutputLength;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in BinFileParams: " << e.what() << std::endl;

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
