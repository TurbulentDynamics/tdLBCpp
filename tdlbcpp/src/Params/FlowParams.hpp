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
using json = nlohmann::json;






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
    bool useLES = false;
    std::string collision = "EgglesSomers";
    std::string streaming = "Esotwist";

    
    
    void calcNuAndRe_m(int impellerBladeOuterRadius){

        Re_m = reMNonDimensional * M_PI / 2.0;

        nu  = uav * (T)impellerBladeOuterRadius / Re_m;
    }

    
    void getParamsFromJson(const json& jsonParams) {


        try
        {

                initialRho = (T)jsonParams["initialRho"].get<double>();
    reMNonDimensional = (T)jsonParams["reMNonDimensional"].get<double>();
    uav = (T)jsonParams["uav"].get<double>();
    cs0 = (T)jsonParams["cs0"].get<double>();
    g3 = (T)jsonParams["g3"].get<double>();
    nu = (T)jsonParams["nu"].get<double>();
    fx0 = (T)jsonParams["fx0"].get<double>();
    fy0 = (T)jsonParams["fy0"].get<double>();
    fz0 = (T)jsonParams["fz0"].get<double>();
    Re_m = (T)jsonParams["Re_m"].get<double>();
    Re_f = (T)jsonParams["Re_f"].get<double>();
    uf = (T)jsonParams["uf"].get<double>();
    alpha = (T)jsonParams["alpha"].get<double>();
    beta = (T)jsonParams["beta"].get<double>();
    useLES = jsonParams["useLES"].get<bool>();
    collision = jsonParams["collision"].get<std::string>();
    streaming = jsonParams["streaming"].get<std::string>();


        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in FlowParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }

    }
    
    
    json getJson() const {

        try {

            json jsonParams;

                jsonParams["initialRho"] = (double)initialRho;
    jsonParams["reMNonDimensional"] = (double)reMNonDimensional;
    jsonParams["uav"] = (double)uav;
    jsonParams["cs0"] = (double)cs0;
    jsonParams["g3"] = (double)g3;
    jsonParams["nu"] = (double)nu;
    jsonParams["fx0"] = (double)fx0;
    jsonParams["fy0"] = (double)fy0;
    jsonParams["fz0"] = (double)fz0;
    jsonParams["Re_m"] = (double)Re_m;
    jsonParams["Re_f"] = (double)Re_f;
    jsonParams["uf"] = (double)uf;
    jsonParams["alpha"] = (double)alpha;
    jsonParams["beta"] = (double)beta;
    jsonParams["useLES"] = useLES;
    jsonParams["collision"] = collision;
    jsonParams["streaming"] = streaming;


            return jsonParams;

        } catch(std::exception& e) {

            std::cerr << "Exception reached parsing arguments in FlowParams: " << e.what() << std::endl;

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
