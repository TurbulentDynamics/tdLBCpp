//
//  CheckpointParams.hpp
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

struct CheckpointParams {

        bool startWithCheckpoint = false;
    std::string checkpointLoadFromDir = "notSet";
    int checkpointRepeat = 10;
    std::string checkpointWriteRootDir = ".";
    std::string checkpointWriteDirPrefix = "debug";
    bool checkpointWriteDirAppendTime = true;

    
    
    bool hasOutputThisStep(tStep step){
        if (checkpointRepeat && (step % checkpointRepeat == 0)) return true;
        return false;
    }

    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {

                startWithCheckpoint = (bool)jsonParams["startWithCheckpoint"].asBool();
    checkpointLoadFromDir = (std::string)jsonParams["checkpointLoadFromDir"].asString();
    checkpointRepeat = (int)jsonParams["checkpointRepeat"].asInt();
    checkpointWriteRootDir = (std::string)jsonParams["checkpointWriteRootDir"].asString();
    checkpointWriteDirPrefix = (std::string)jsonParams["checkpointWriteDirPrefix"].asString();
    checkpointWriteDirAppendTime = (bool)jsonParams["checkpointWriteDirAppendTime"].asBool();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in CheckpointParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;
            
                jsonParams["startWithCheckpoint"] = (bool)startWithCheckpoint;
    jsonParams["checkpointLoadFromDir"] = (std::string)checkpointLoadFromDir;
    jsonParams["checkpointRepeat"] = (int)checkpointRepeat;
    jsonParams["checkpointWriteRootDir"] = (std::string)checkpointWriteRootDir;
    jsonParams["checkpointWriteDirPrefix"] = (std::string)checkpointWriteDirPrefix;
    jsonParams["checkpointWriteDirAppendTime"] = (bool)checkpointWriteDirAppendTime;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in CheckpointParams: " << e.what() << std::endl;

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
            
            std::cerr << "Exception writing json file for CheckpointParams: " << e.what() << std::endl;
        }
        
        return 0;
    }
    
    
    void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
