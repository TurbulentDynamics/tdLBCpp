
//
//  OutputData.hpp.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Niall OByrnes on 23/07/2018.
//  Copyright © 2018 Niall P. O'Byrnes. All rights reserved.
//

#ifndef Output_Data_hpp
#define Output_Data_hpp

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "json.h"




struct Plane {
    std::string name_root;
    tStep repeat = 0;
    tNi cutAt = 0;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float = 0;
    
    std::string QDataType;

    
    Plane getParamFromJson(const std::string filePath){
        
        Plane g;
        
        try
        {
            std::ifstream in(filePath.c_str());
            Json::Value jsonParams;
            in >> jsonParams;
            
            g.name_root = jsonParams["name_root"].asString();
            g.repeat = jsonParams["repeat"].asInt();
            g.cutAt = jsonParams["cutAt"].asInt();

            g.Q_output_len = jsonParams["Q_output_len"].asInt();
            g.start_at_step = (tNi)jsonParams["start_at_step"].asInt();

            g.end_at_repeat = jsonParams["end_at_repeat"].asInt();
            g.use_half_float = jsonParams["use_half_float"].asBool();
            
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
    
    
    

    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;
            jsonParams["name"] = "Plane";
            
            jsonParams["name_root"] = name_root;
            jsonParams["repeat"] = (int)repeat;
            jsonParams["cutAt"] = (int)cutAt;
            
            jsonParams["Q_output_len"] = (int)Q_output_len;
            jsonParams["start_at_step"] = (int)start_at_step;
            jsonParams["end_at_repeat"] = (int)end_at_repeat;
            jsonParams["use_half_float"] = (bool)use_half_float;
            
            jsonParams["QDataType"] = (std::string)QDataType;
            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return "";
        }
    }
        
        
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


};



















struct Volume {
    std::string name_root;
    tStep repeat = 0;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;
};






struct Angle {
    std::string name_root;
    tStep repeat = 0;
    double degrees = 0;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;
};




struct PLaneAtAngle {
    std::string name_root;
    double degrees = 0;
    double tolerance = 0;
    tNi cutAt = 0;
    
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;
};



struct Sector {
    std::string name_root;
    tStep repeat = 0;
    
    
    double angle_infront_blade = 0.0f;
    double angle_behind_blade = 0.0f;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;
};





#endif
