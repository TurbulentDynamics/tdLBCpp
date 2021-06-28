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
    std::string checkpoint_root_dir = "";


    CheckpointParams getParamFromJson(const std::string filePath){
        
        CheckpointParams g;
        
        try
        {
            std::ifstream in(filePath.c_str());
            Json::Value jsonParams;
            in >> jsonParams;
            
            g.start_with_checkpoint = jsonParams["start_with_checkpoint"].asBool();
            g.load_checkpoint_dirname = jsonParams["load_checkpoint_dirname"].asString();
            
            g.checkpoint_repeat = jsonParams["checkpoint_repeat"].asInt();
            g.checkpoint_root_dir = jsonParams["checkpoint_root_dir"].asString();
           
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
    
};//end of struct


#endif /* Checkpoint_h */
