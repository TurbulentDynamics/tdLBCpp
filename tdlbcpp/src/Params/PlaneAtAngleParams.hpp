//
//  PlaneAtAngleParams.hpp
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

struct PlaneAtAngleParams {

        std::string name_root = "plane_at_angle";
        double degrees = 0.0;
        double tolerance = 0.0;
        tNi cutAt = 0;
        int Q_output_len = 4;
        tStep start_at_step = 0;
        tStep end_at_step = 0;
        bool use_half_float = false;
        std::string QDataType = "float";

    
    
    
        void getParamsFromJson(const json& jsonParams) {

        
        try
        {

                name_root = jsonParams["name_root"].get<std::string>();
        degrees = (double)jsonParams["degrees"].get<double>();
        tolerance = (double)jsonParams["tolerance"].get<double>();
        cutAt = (tNi)jsonParams["cutAt"].get<uint64_t>();
        Q_output_len = (int)jsonParams["Q_output_len"].get<int>();
        start_at_step = (tStep)jsonParams["start_at_step"].get<uint64_t>();
        end_at_step = (tStep)jsonParams["end_at_step"].get<uint64_t>();
        use_half_float = jsonParams["use_half_float"].get<bool>();
        QDataType = jsonParams["QDataType"].get<std::string>();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in PlaneAtAngleParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
        json getJson() const {
        
        try {
            
            json jsonParams;
            
                jsonParams["name_root"] = (std::string)name_root;
        jsonParams["degrees"] = (double)degrees;
        jsonParams["tolerance"] = (double)tolerance;
        jsonParams["cutAt"] = cutAt;
        jsonParams["Q_output_len"] = (int)Q_output_len;
        jsonParams["start_at_step"] = start_at_step;
        jsonParams["end_at_step"] = end_at_step;
        jsonParams["use_half_float"] = (bool)use_half_float;
        jsonParams["QDataType"] = (std::string)QDataType;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in PlaneAtAngleParams: " << e.what() << std::endl;

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
            
            std::cerr << "Exception writing json file for PlaneAtAngleParams: " << e.what() << std::endl;
        }
        
        return 0;
    }
    
    
        void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
