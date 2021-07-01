//
//  Checkpoint.h
//  tdLBCpp
//
//  Created by Niall Ó Broin on 28/06/2021.
//

#ifndef Checkpoint_h
#define Checkpoint_h

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "json.h"


struct CheckpointParams
{

    bool start_with_checkpoint = false;
    std::string load_checkpoint_dirname = "";
    
    int checkpoint_repeat = 0;
    std::string checkpoint_root_dir = "checkpoint_root_dir";

    
    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {
            start_with_checkpoint = jsonParams["start_with_checkpoint"].asBool();
            load_checkpoint_dirname = jsonParams["load_checkpoint_dirname"].asString();
            
            checkpoint_repeat = jsonParams["checkpoint_repeat"].asInt();
            checkpoint_root_dir = jsonParams["checkpoint_root_dir"].asString();
            
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
        }
    };
    
    

    
    
    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;

            jsonParams["start_with_checkpoint"] = start_with_checkpoint;
            jsonParams["load_checkpoint_dirname"] = load_checkpoint_dirname;
            
            jsonParams["checkpoint_repeat"] = checkpoint_repeat;
            jsonParams["checkpoint_root_dir"] = checkpoint_root_dir;
            
            
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


#endif /* Checkpoint_h */
