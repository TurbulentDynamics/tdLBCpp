//
//  SectorParams.hpp
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

struct SectorParams {

        std::string name_root = "sector";
        tStep repeat = 0;
        double angle_infront_blade = 0.0;
        double angle_behind_blade = 0.0;
        int Q_output_len = 4;
        tStep start_at_step = 0;
        tStep end_at_step = 0;
        bool use_half_float = false;
        std::string QDataType = "float";

    
    
    
        void getParamsFromJson(const json& jsonParams) {

        
        try
        {

                name_root = jsonParams["name_root"].get<std::string>();
        repeat = (tStep)jsonParams["repeat"].get<uint64_t>();
        angle_infront_blade = (double)jsonParams["angle_infront_blade"].get<double>();
        angle_behind_blade = (double)jsonParams["angle_behind_blade"].get<double>();
        Q_output_len = (int)jsonParams["Q_output_len"].get<int>();
        start_at_step = (tStep)jsonParams["start_at_step"].get<uint64_t>();
        end_at_step = (tStep)jsonParams["end_at_step"].get<uint64_t>();
        use_half_float = jsonParams["use_half_float"].get<bool>();
        QDataType = jsonParams["QDataType"].get<std::string>();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in SectorParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
        json getJson() const {
        
        try {
            
            json jsonParams;
            
                jsonParams["name_root"] = (std::string)name_root;
        jsonParams["repeat"] = repeat;
        jsonParams["angle_infront_blade"] = (double)angle_infront_blade;
        jsonParams["angle_behind_blade"] = (double)angle_behind_blade;
        jsonParams["Q_output_len"] = (int)Q_output_len;
        jsonParams["start_at_step"] = start_at_step;
        jsonParams["end_at_step"] = end_at_step;
        jsonParams["use_half_float"] = (bool)use_half_float;
        jsonParams["QDataType"] = (std::string)QDataType;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in SectorParams: " << e.what() << std::endl;

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
            
            std::cerr << "Exception writing json file for SectorParams: " << e.what() << std::endl;
        }
        
        return 0;
    }
    
    
        void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
