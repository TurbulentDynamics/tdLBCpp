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




struct BinFileParams
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
    
    

    void getParamsFromJson(Json::Value jsonParams) {

        try
        {

            filePath = jsonParams["filePath"].asString();
            name = jsonParams["name"].asString();
            note = jsonParams["note"].asString();

            structName = jsonParams["structName"].asString();
            binFileSizeInStructs = (tNi)jsonParams["binFileSizeInStructs"].asUInt64();

            coordsType = jsonParams["coordsType"].asString();
            hasGridtCoords = jsonParams["hasGridtCoords"].asBool();
            hasColRowtCoords = jsonParams["hasColRowtCoords"].asBool();

            
            QDataType = jsonParams["QDataType"].asString();
            QOutputLength = jsonParams["QOutputLength"].asInt();
           
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
        }
                
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

            jsonParams["filePath"] = filePath;
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
