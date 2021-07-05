//
//  define_datastructures.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Niall Ó Broin on 08/01/2019.
//  Copyright © 2019 Niall Ó Broin. All rights reserved.
//

#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "json.h"




template <typename T>
struct FlowParams {
    
    T initialRho = (T)8.0;
    T reMNonDimensional = (T)07000.0;
    
    T uav = (T)0.1;
 
    T cs0 = 0.0;
    T g3 = (T)0.8;
    
    T nu = 0.0;
    
    T fx0 = 0.0;
    
    T Re_m = 0.0;
    T Re_f = 0.0;
    T uf = 0.0;
    
    T alpha = (T)0.97;
    T beta = (T)1.9;

    bool useLES = 0;
    
    std::string collision = "EgglesSomers";
    std::string streaming = "Nieve";
    
    
    
    FlowParams<double> asDouble(){
        FlowParams<double> f;

        f.initialRho = (double)initialRho;
        f.reMNonDimensional = (double)reMNonDimensional;
        f.uav = (double)uav;
        f.cs0 = (double)cs0;
        f.g3 = (double)g3;
        f.nu = (double)nu;
        f.fx0 = (double)fx0;
        f.Re_m = (double)Re_m;
        f.Re_f = (double)Re_f;
        f.uf = (double)uf;
        
        f.alpha = (double)alpha;
        f.beta = (double)beta;

        f.useLES = useLES;
        f.collision = collision;
        f.streaming = streaming;

        return f;
    }
        


    void getParamsFromJson(Json::Value jsonParams) {

        try
        {
            
            initialRho = (T)jsonParams["initialRho"].asDouble();
            reMNonDimensional = (T)jsonParams["reMNonDimensional"].asDouble();

            uav = (T)jsonParams["uav"].asDouble();

            cs0 = (T)jsonParams["cs0"].asDouble();
            g3 = (T)jsonParams["g3"].asDouble();

            nu = (T)jsonParams["nu"].asDouble();
            fx0 = (T)jsonParams["fx0"].asDouble();

            Re_m = (T)jsonParams["Re_m"].asDouble();
            Re_f = (T)jsonParams["Re_f"].asDouble();
            uf = (T)jsonParams["uf"].asDouble();

            alpha = (T)jsonParams["alpha"].asDouble();
            beta = (T)jsonParams["beta"].asDouble();
            
            useLES = (T)jsonParams["useLES"].asBool();
            collision = jsonParams["collision"].asString();
            streaming = jsonParams["streaming"].asString();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
        }
    }
    
    

    Json::Value getJson(){
        try
        {
            Json::Value jsonParams;
            
            jsonParams["name"] = "FlowParams";
            
            jsonParams["initialRho"] = (double)initialRho;
            jsonParams["reMNonDimensional"] = (double)reMNonDimensional;
            jsonParams["uav"] = (double)uav;
            jsonParams["cs0"] = (double)cs0;
            jsonParams["nu"] = (double)nu;
            jsonParams["g3"] = (double)g3;
            jsonParams["fx0"] = (double)fx0;
            jsonParams["Re_f"] = (double)Re_f;
            jsonParams["uf"] = (double)uf;
            jsonParams["Re_m"] = (double)Re_m;
            
            jsonParams["alpha"] = (double)alpha;
            jsonParams["beta"] = (double)beta;

            jsonParams["useLES"] = (bool)useLES;
            jsonParams["collision"] = (std::string)collision;
            jsonParams["streaming"] = (std::string)streaming;

            
            
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
    
    
    
}; //end of struct
