//
//  FlowParams.hpp
//  Turbulent Dynamics Lattice Boltzmann Cpp
//
//  Created by Niall Ó Broin on 08/01/2019.
//  Copyright © 2019 Niall Ó Broin. All rights reserved.
//

#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cmath>

#include <nlohmann/json.hpp>






//
template <typename T>
struct FlowParams {

        T initialRho = 8.0;
        T reMNonDimensional = 7300.0;
        T uav = 0.1;

    //ratio mixing length / lattice spacing delta (Smagorinsky)
        T cs0 = 0.12;

    //compensation of third order terms
        T g3 = 0.8;

    //kinematic viscosity
        T nu = 0.0;

    //forcing in x-direction
        T fx0 = 0.0;

    //forcing in y-direction
        T fy0 = 0.0;

    //forcing in z-direction
        T fz0 = 0.0;

    //Reynolds number based on mean or tip velocity
        T Re_m = 0.0;

    //Reynolds number based on the friction velocity uf
        T Re_f = 0.0;

    //friction velocity
        T uf = 0.0;
        T alpha = 0.97;
        T beta = 1.9;
        std::string collision = "EgglesSomers";
        std::string streaming = "Esotwist";

    
    
        void calcNuAndRe_m(int impellerBladeOuterRadius){

        Re_m = reMNonDimensional * M_PI / 2.0;

        nu  = uav * static_cast<T>(impellerBladeOuterRadius) / Re_m;
    }

    
        void getParamsFromJson(const nlohmann::json& jsonParams) {


        try
        {

                initialRho = static_cast<T>(jsonParams["initialRho"].get<double>());
        reMNonDimensional = static_cast<T>(jsonParams["reMNonDimensional"].get<double>());
        uav = static_cast<T>(jsonParams["uav"].get<double>());
        cs0 = static_cast<T>(jsonParams["cs0"].get<double>());
        g3 = static_cast<T>(jsonParams["g3"].get<double>());
        nu = static_cast<T>(jsonParams["nu"].get<double>());
        fx0 = static_cast<T>(jsonParams["fx0"].get<double>());
        fy0 = static_cast<T>(jsonParams["fy0"].get<double>());
        fz0 = static_cast<T>(jsonParams["fz0"].get<double>());
        Re_m = static_cast<T>(jsonParams["Re_m"].get<double>());
        Re_f = static_cast<T>(jsonParams["Re_f"].get<double>());
        uf = static_cast<T>(jsonParams["uf"].get<double>());
        alpha = static_cast<T>(jsonParams["alpha"].get<double>());
        beta = static_cast<T>(jsonParams["beta"].get<double>());
        collision = jsonParams["collision"].get<std::string>();
        streaming = jsonParams["streaming"].get<std::string>();


        }
        catch(const nlohmann::json::exception& e)
        {
            std::cerr << "JSON parsing error in FlowParams:: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Failed to parse FlowParams: ") + e.what());
        }

    }
    
    
        nlohmann::json getJson() const {

        try {

            nlohmann::json jsonParams;

                jsonParams["initialRho"] = static_cast<double>(initialRho);
        jsonParams["reMNonDimensional"] = static_cast<double>(reMNonDimensional);
        jsonParams["uav"] = static_cast<double>(uav);
        jsonParams["cs0"] = static_cast<double>(cs0);
        jsonParams["g3"] = static_cast<double>(g3);
        jsonParams["nu"] = static_cast<double>(nu);
        jsonParams["fx0"] = static_cast<double>(fx0);
        jsonParams["fy0"] = static_cast<double>(fy0);
        jsonParams["fz0"] = static_cast<double>(fz0);
        jsonParams["Re_m"] = static_cast<double>(Re_m);
        jsonParams["Re_f"] = static_cast<double>(Re_f);
        jsonParams["uf"] = static_cast<double>(uf);
        jsonParams["alpha"] = static_cast<double>(alpha);
        jsonParams["beta"] = static_cast<double>(beta);
        jsonParams["collision"] = collision;
        jsonParams["streaming"] = streaming;


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
            throw std::runtime_error(std::string("Failed to parse FlowParams: ") + e.what());
        }

    };
    
    
    
    
        int writeParamsToJsonFile(const std::string filePath) {


        try {

            nlohmann::json jsonParams = getJson();

            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams.dump(4);  // Pretty print with 4 spaces
            out.close();

        } catch(const nlohmann::json::exception& e){

            std::cerr << "Exception writing json file for FlowParams: " << e.what() << std::endl;
        }

        return 0;
    }
    
    
        void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
