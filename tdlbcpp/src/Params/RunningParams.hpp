//
//  RunningParams.hpp
//  Turbulent Dynamics Lattice Boltzmann Cpp
//
//  Created by Niall Ó Broin on 08/01/2019.
//  Copyright © 2019 Niall Ó Broin. All rights reserved.
//

#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "json.h"






//

struct RunningParams {

        tStep step = 1;
    double angle = 0.0;
    tStep num_steps = 20;
    tStep impellerStartupStepsUntilNormalSpeed = 30;
    tStep repeatCheckForErrors = 100;
    tStep numStepsForAverageCalc = 10;
    tStep repeatPrintTimerToFile = 20;
    tStep repeatPrintTimerToStdOut = 10;
    std::string runningDataFileDir = ".";
    std::string runningDataFilePrefix = "debug";
    bool runningDataFileAppendTime = true;
    tStep doubleResolutionAtStep = 10;

    
    
    void update(tStep _step, double _angle){

        step = (tStep)_step;
        angle = (double)_angle;

    }

    void incrementStep(){
        step ++;
    }

    
    void getParamsFromJson(Json::Value jsonParams) {

        
        try
        {

                step = (tStep)jsonParams["step"].asUInt64();
    angle = (double)jsonParams["angle"].asDouble();
    num_steps = (tStep)jsonParams["num_steps"].asUInt64();
    impellerStartupStepsUntilNormalSpeed = (tStep)jsonParams["impellerStartupStepsUntilNormalSpeed"].asUInt64();
    repeatCheckForErrors = (tStep)jsonParams["repeatCheckForErrors"].asUInt64();
    numStepsForAverageCalc = (tStep)jsonParams["numStepsForAverageCalc"].asUInt64();
    repeatPrintTimerToFile = (tStep)jsonParams["repeatPrintTimerToFile"].asUInt64();
    repeatPrintTimerToStdOut = (tStep)jsonParams["repeatPrintTimerToStdOut"].asUInt64();
    runningDataFileDir = (std::string)jsonParams["runningDataFileDir"].asString();
    runningDataFilePrefix = (std::string)jsonParams["runningDataFilePrefix"].asString();
    runningDataFileAppendTime = (bool)jsonParams["runningDataFileAppendTime"].asBool();
    doubleResolutionAtStep = (tStep)jsonParams["doubleResolutionAtStep"].asUInt64();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in RunningParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
    Json::Value getJson(){
        
        try {
            
            Json::Value jsonParams;
            
                jsonParams["step"] = (Json::UInt64)step;
    jsonParams["angle"] = (double)angle;
    jsonParams["num_steps"] = (Json::UInt64)num_steps;
    jsonParams["impellerStartupStepsUntilNormalSpeed"] = (Json::UInt64)impellerStartupStepsUntilNormalSpeed;
    jsonParams["repeatCheckForErrors"] = (Json::UInt64)repeatCheckForErrors;
    jsonParams["numStepsForAverageCalc"] = (Json::UInt64)numStepsForAverageCalc;
    jsonParams["repeatPrintTimerToFile"] = (Json::UInt64)repeatPrintTimerToFile;
    jsonParams["repeatPrintTimerToStdOut"] = (Json::UInt64)repeatPrintTimerToStdOut;
    jsonParams["runningDataFileDir"] = (std::string)runningDataFileDir;
    jsonParams["runningDataFilePrefix"] = (std::string)runningDataFilePrefix;
    jsonParams["runningDataFileAppendTime"] = (bool)runningDataFileAppendTime;
    jsonParams["doubleResolutionAtStep"] = (Json::UInt64)doubleResolutionAtStep;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in RunningParams: " << e.what() << std::endl;

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
            std::cerr << "Exception reading from input file: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
        
    };
    
    
    
    
    int writeParamsToJsonFile(const std::string filePath) {
        
        
        try {
            
            Json::Value jsonParams = getJson();
            
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams;
            out.close();
            
        } catch(std::exception& e){
            
            std::cerr << "Exception writing json file for RunningParams: " << e.what() << std::endl;
        }
        
        return 0;
    }
    
    
    void printParams() {
        
        std::cout
        << getJson()
        << std::endl;
        
    }
    
};//end of struct
