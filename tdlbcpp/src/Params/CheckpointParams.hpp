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

        bool start_with_checkpoint = false;
    std::string load_checkpoint_dirname = "debug_checkpoint_dir/debug_checkpoint";
    int checkpoint_repeat = 0;
    std::string checkpointRootDir = "debug_checkpoint_dir";

    
    
    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {

                start_with_checkpoint = (bool)jsonParams["start_with_checkpoint"].asBool();
    load_checkpoint_dirname = (std::string)jsonParams["load_checkpoint_dirname"].asString();
    checkpoint_repeat = (int)jsonParams["checkpoint_repeat"].asInt();
    checkpointRootDir = (std::string)jsonParams["checkpointRootDir"].asString();

            
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
            
                jsonParams["start_with_checkpoint"] = (bool)start_with_checkpoint;
    jsonParams["load_checkpoint_dirname"] = (std::string)load_checkpoint_dirname;
    jsonParams["checkpoint_repeat"] = (int)checkpoint_repeat;
    jsonParams["checkpointRootDir"] = (std::string)checkpointRootDir;

            
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
