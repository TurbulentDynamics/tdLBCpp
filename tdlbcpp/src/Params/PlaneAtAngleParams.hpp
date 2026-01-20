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

    
    
    
        void getParamsFromJson(const nlohmann::json& jsonParams) {

        
        try
        {

                name_root = jsonParams["name_root"].get<std::string>();
        degrees = static_cast<double>(jsonParams["degrees"].get<double>());
        tolerance = static_cast<double>(jsonParams["tolerance"].get<double>());
        cutAt = (tNi)jsonParams["cutAt"].get<uint64_t>();
        Q_output_len = static_cast<int>(jsonParams["Q_output_len"].get<int>());
        start_at_step = jsonParams["start_at_step"].get<tStep>();
        end_at_step = jsonParams["end_at_step"].get<tStep>();
        use_half_float = jsonParams["use_half_float"].get<bool>();
        QDataType = jsonParams["QDataType"].get<std::string>();

            
        }
        catch(const nlohmann::json::exception& e)
        {
            std::cerr << "JSON parsing error in PlaneAtAngleParams:: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Failed to parse PlaneAtAngleParams: ") + e.what());
        }
                
    }
    
    
        nlohmann::json getJson() const {
        
        try {
            
            nlohmann::json jsonParams;
            
                jsonParams["name_root"] = static_cast<std::string>(name_root);
        jsonParams["degrees"] = static_cast<double>(degrees);
        jsonParams["tolerance"] = static_cast<double>(tolerance);
        jsonParams["cutAt"] = cutAt;
        jsonParams["Q_output_len"] = static_cast<int>(Q_output_len);
        jsonParams["start_at_step"] = start_at_step;
        jsonParams["end_at_step"] = end_at_step;
        jsonParams["use_half_float"] = static_cast<bool>(use_half_float);
        jsonParams["QDataType"] = static_cast<std::string>(QDataType);

            
            return jsonParams;
            
        } catch(const nlohmann::json::exception& e) {
            
            std::cerr << "JSON parsing error in PlaneAtAngleParams:: " << e.what() << std::endl;

            return nlohmann::json();
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
            throw std::runtime_error(std::string("Failed to parse PlaneAtAngleParams: ") + e.what());
        }
        
    };
    
    
    
    
        int writeParamsToJsonFile(const std::string filePath) {
        
        
        try {
            
            nlohmann::json jsonParams = getJson();
            
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams.dump(4);  // Pretty print with 4 spaces
            out.close();
            
        } catch(const nlohmann::json::exception& e){
            
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
