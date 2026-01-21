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






//

struct CheckpointParams {

        bool startWithCheckpoint = false;
        std::string checkpointLoadFromDir = "notSet";
        int checkpointRepeat = 10;
        std::string checkpointWriteRootDir = ".";
        std::string checkpointWriteDirPrefix = "debug";
        bool checkpointWriteDirAppendTime = true;

    
    
    
        void getParamsFromJson(const nlohmann::json& jsonParams) {

        
        try
        {

                startWithCheckpoint = jsonParams["startWithCheckpoint"].get<bool>();
        checkpointLoadFromDir = jsonParams["checkpointLoadFromDir"].get<std::string>();
        checkpointRepeat = static_cast<int>(jsonParams["checkpointRepeat"].get<int>());
        checkpointWriteRootDir = jsonParams["checkpointWriteRootDir"].get<std::string>();
        checkpointWriteDirPrefix = jsonParams["checkpointWriteDirPrefix"].get<std::string>();
        checkpointWriteDirAppendTime = jsonParams["checkpointWriteDirAppendTime"].get<bool>();

            
        }
        catch(const nlohmann::json::exception& e)
        {
            std::cerr << "JSON parsing error in CheckpointParams:: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Failed to parse CheckpointParams: ") + e.what());
        }
                
    }
    
    
        nlohmann::json getJson() const {
        
        try {
            
            nlohmann::json jsonParams;
            
                jsonParams["startWithCheckpoint"] = static_cast<bool>(startWithCheckpoint);
        jsonParams["checkpointLoadFromDir"] = static_cast<std::string>(checkpointLoadFromDir);
        jsonParams["checkpointRepeat"] = static_cast<int>(checkpointRepeat);
        jsonParams["checkpointWriteRootDir"] = static_cast<std::string>(checkpointWriteRootDir);
        jsonParams["checkpointWriteDirPrefix"] = static_cast<std::string>(checkpointWriteDirPrefix);
        jsonParams["checkpointWriteDirAppendTime"] = static_cast<bool>(checkpointWriteDirAppendTime);

            
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
            throw std::runtime_error(std::string("Failed to parse CheckpointParams: ") + e.what());
        }
        
    };
    
    
    
    
        int writeParamsToJsonFile(const std::string filePath) {
        
        
        try {
            
            nlohmann::json jsonParams = getJson();
            
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams.dump(4);  // Pretty print with 4 spaces
            out.close();
            
        } catch(const nlohmann::json::exception& e){
            
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
