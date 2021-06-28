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




struct BinFileFormat
{

    
    std::string filePath = "";
    std::string name = "";
    std::string note = "";

    
    std::string structName = "";
    tNi binFileSizeInStructs = 0;
    
    std::string coordsType = "";
    bool hasGridtCoords;
    bool hasColRowtCoords;

    std::string QDataType = "";
    int QOutputLength = 0;
    
    

    BinFileFormat getParamFromJson(const std::string filePath){
        
        BinFileFormat g;
        
        try
        {
            std::ifstream in(filePath.c_str());
            Json::Value jsonParams;
            in >> jsonParams;
            
            g.filePath = jsonParams["filePath"].asString();
            g.name = jsonParams["name"].asString();
            g.note = jsonParams["note"].asString();

            g.structName = jsonParams["structName"].asString();
            g.binFileSizeInStructs = (tNi)jsonParams["binFileSizeInStructs"].asUInt64();

            g.coordsType = jsonParams["coordsType"].asString();
            g.hasGridtCoords = jsonParams["hasGridtCoords"].asBool();
            g.hasColRowtCoords = jsonParams["hasColRowtCoords"].asBool();

            
            g.QDataType = jsonParams["QDataType"].asString();
            g.QOutputLength = jsonParams["QOutputLength"].asInt();
           
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

            jsonParams["filePath"] = note;
            jsonParams["name"] = name;
            jsonParams["note"] = note;
            
            
            
            jsonParams["structName"] = structName;
            jsonParams["binFileSizeInStructs"] = (uint64_t) binFileSizeInStructs;
            
            
            jsonParams["coordsType"] = coordsType;
            jsonParams["hasGridtCoords"] = hasGridtCoords;
            jsonParams["hasColRowtCoords"] = hasColRowtCoords;
            
            
            jsonParams["QDataType"] = QDataType;
            jsonParams["QOutputLength"] = (int)QOutputLength;

        

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return "";
        }
    }
    
    void print(){
        
        std::cout
        << " name:" << "BinFileParams"
        << " filePath:" << filePath
        << " note:" << note

        
        << " structName:" << structName
        << " binFileSizeInStructs:" << binFileSizeInStructs
        
        << " coordsType:" << coordsType
        << " hasGridtCoords:" << hasGridtCoords
        << " hasColRowtCoords:" << hasColRowtCoords

        << " QDataType:" << QDataType
        << " QOutputLength:" << QOutputLength
        
        << std::endl;
        
    }
    
};//end of struct
