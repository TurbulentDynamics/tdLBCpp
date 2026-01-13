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

#include <nlohmann/json.hpp>
using json = nlohmann::json;






//

struct CheckpointParams {

        bool startWithCheckpoint = false;
        std::string checkpointLoadFromDir = "notSet";
        int checkpointRepeat = 10;
        std::string checkpointWriteRootDir = ".";
        std::string checkpointWriteDirPrefix = "debug";
        bool checkpointWriteDirAppendTime = true;

    
    
    
        void getParamsFromJson(const json& jsonParams) {

        
        try
        {

                startWithCheckpoint = jsonParams["startWithCheckpoint"].get<bool>();
        checkpointLoadFromDir = jsonParams["checkpointLoadFromDir"].get<std::string>();
        checkpointRepeat = (int)jsonParams["checkpointRepeat"].get<int>();
        checkpointWriteRootDir = jsonParams["checkpointWriteRootDir"].get<std::string>();
        checkpointWriteDirPrefix = jsonParams["checkpointWriteDirPrefix"].get<std::string>();
        checkpointWriteDirAppendTime = jsonParams["checkpointWriteDirAppendTime"].get<bool>();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in CheckpointParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
        json getJson() const {
        
        try {
            
            json jsonParams;
            
                jsonParams["startWithCheckpoint"] = (bool)startWithCheckpoint;
        jsonParams["checkpointLoadFromDir"] = (std::string)checkpointLoadFromDir;
        jsonParams["checkpointRepeat"] = (int)checkpointRepeat;
        jsonParams["checkpointWriteRootDir"] = (std::string)checkpointWriteRootDir;
        jsonParams["checkpointWriteDirPrefix"] = (std::string)checkpointWriteDirPrefix;
        jsonParams["checkpointWriteDirAppendTime"] = (bool)checkpointWriteDirAppendTime;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in CheckpointParams: " << e.what() << std::endl;

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
