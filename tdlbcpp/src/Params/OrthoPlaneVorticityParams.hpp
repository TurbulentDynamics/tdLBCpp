//
//  OrthoPlaneVorticityParams.hpp
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

struct OrthoPlaneVorticityParams {

        std::string name_root = "vort";
        int jpegCompression = 100;
        tNi cutAt = 0;
        tStep repeat = 0;
        tStep start_at_step = 0;
        tStep end_at_step = 0;

    
    
    
        void getParamsFromJson(const nlohmann::json& jsonParams) {

        
        try
        {

                name_root = jsonParams["name_root"].get<std::string>();
        jpegCompression = static_cast<int>(jsonParams["jpegCompression"].get<int>());
        cutAt = (tNi)jsonParams["cutAt"].get<uint64_t>();
        repeat = jsonParams["repeat"].get<tStep>();
        start_at_step = jsonParams["start_at_step"].get<tStep>();
        end_at_step = jsonParams["end_at_step"].get<tStep>();

            
        }
        catch(const nlohmann::json::exception& e)
        {
            std::cerr << "JSON parsing error in OrthoPlaneVorticityParams:: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Failed to parse OrthoPlaneVorticityParams: ") + e.what());
        }
                
    }
    
    
        nlohmann::json getJson() const {
        
        try {
            
            nlohmann::json jsonParams;
            
                jsonParams["name_root"] = static_cast<std::string>(name_root);
        jsonParams["jpegCompression"] = static_cast<int>(jpegCompression);
        jsonParams["cutAt"] = cutAt;
        jsonParams["repeat"] = repeat;
        jsonParams["start_at_step"] = start_at_step;
        jsonParams["end_at_step"] = end_at_step;

            
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
            throw std::runtime_error(std::string("Failed to parse OrthoPlaneVorticityParams: ") + e.what());
        }
        
    };
    
    
    
    
        int writeParamsToJsonFile(const std::string filePath) {
        
        
        try {
            
            nlohmann::json jsonParams = getJson();
            
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams.dump(4);  // Pretty print with 4 spaces
            out.close();
            
        } catch(const nlohmann::json::exception& e){
            
            std::cerr << "Exception writing json file for OrthoPlaneVorticityParams: " << e.what() << std::endl;
        }
        
        return 0;
    }
    
    
        void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
