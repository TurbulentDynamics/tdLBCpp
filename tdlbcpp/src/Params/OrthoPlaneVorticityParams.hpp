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
using json = nlohmann::json;






//

struct OrthoPlaneVorticityParams {

        std::string name_root = "vort";
    int jpegCompression = 100;
    tNi cutAt = 0;
    tStep repeat = 0;
    tStep start_at_step = 0;
    tStep end_at_step = 0;

    
    
    
    void getParamsFromJson(const json& jsonParams) {

        
        try
        {

                name_root = jsonParams["name_root"].get<std::string>();
    jpegCompression = (int)jsonParams["jpegCompression"].get<int>();
    cutAt = (tNi)jsonParams["cutAt"].get<uint64_t>();
    repeat = (tStep)jsonParams["repeat"].get<uint64_t>();
    start_at_step = (tStep)jsonParams["start_at_step"].get<uint64_t>();
    end_at_step = (tStep)jsonParams["end_at_step"].get<uint64_t>();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in OrthoPlaneVorticityParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
    json getJson() const {
        
        try {
            
            json jsonParams;
            
                jsonParams["name_root"] = (std::string)name_root;
    jsonParams["jpegCompression"] = (int)jpegCompression;
    jsonParams["cutAt"] = cutAt;
    jsonParams["repeat"] = repeat;
    jsonParams["start_at_step"] = start_at_step;
    jsonParams["end_at_step"] = end_at_step;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in OrthoPlaneVorticityParams: " << e.what() << std::endl;

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
