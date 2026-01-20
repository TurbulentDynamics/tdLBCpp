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

#include <nlohmann/json.hpp>






//

struct RunningParams {

        tStep step = 1;
        double angle = 0.0;
        tStep num_steps = 20;
        tStep impellerStartupStepsUntilNormalSpeed = 30;
        tStep numStepsForAverageCalc = 10;
        tStep repeatPrintTimerToFile = 20;
        tStep repeatPrintTimerToStdOut = 10;
        std::string runningDataFileDir = ".";
        std::string runningDataFilePrefix = "debug";
        bool runningDataFileAppendTime = true;
        tStep doubleResolutionAtStep = 10;

    
    
        void update(tStep _step, double _angle){

        step = (tStep)_step;
        angle = static_cast<double>(_angle);

    }

        void incrementStep(){
        step ++;
    }

    
        void getParamsFromJson(const nlohmann::json& jsonParams) {

        
        try
        {

                step = (tStep)jsonParams["step"].get<uint64_t>();
        angle = static_cast<double>(jsonParams["angle"].get<double>());
        num_steps = (tStep)jsonParams["num_steps"].get<uint64_t>();
        impellerStartupStepsUntilNormalSpeed = (tStep)jsonParams["impellerStartupStepsUntilNormalSpeed"].get<uint64_t>();
        numStepsForAverageCalc = (tStep)jsonParams["numStepsForAverageCalc"].get<uint64_t>();
        repeatPrintTimerToFile = (tStep)jsonParams["repeatPrintTimerToFile"].get<uint64_t>();
        repeatPrintTimerToStdOut = (tStep)jsonParams["repeatPrintTimerToStdOut"].get<uint64_t>();
        runningDataFileDir = jsonParams["runningDataFileDir"].get<std::string>();
        runningDataFilePrefix = jsonParams["runningDataFilePrefix"].get<std::string>();
        runningDataFileAppendTime = jsonParams["runningDataFileAppendTime"].get<bool>();
        doubleResolutionAtStep = (tStep)jsonParams["doubleResolutionAtStep"].get<uint64_t>();

            
        }
        catch(std::exception& e)
        {
            std::cerr << "Exception reached parsing arguments in RunningParams: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
                
    }
    
    
        nlohmann::json getJson() const {
        
        try {
            
            nlohmann::json jsonParams;
            
                jsonParams["step"] = step;
        jsonParams["angle"] = static_cast<double>(angle);
        jsonParams["num_steps"] = num_steps;
        jsonParams["impellerStartupStepsUntilNormalSpeed"] = impellerStartupStepsUntilNormalSpeed;
        jsonParams["numStepsForAverageCalc"] = numStepsForAverageCalc;
        jsonParams["repeatPrintTimerToFile"] = repeatPrintTimerToFile;
        jsonParams["repeatPrintTimerToStdOut"] = repeatPrintTimerToStdOut;
        jsonParams["runningDataFileDir"] = static_cast<std::string>(runningDataFileDir);
        jsonParams["runningDataFilePrefix"] = static_cast<std::string>(runningDataFilePrefix);
        jsonParams["runningDataFileAppendTime"] = static_cast<bool>(runningDataFileAppendTime);
        jsonParams["doubleResolutionAtStep"] = doubleResolutionAtStep;

            
            return jsonParams;
            
        } catch(std::exception& e) {
            
            std::cerr << "Exception reached parsing arguments in RunningParams: " << e.what() << std::endl;

            return json();
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
        catch(std::exception& e)
        {
            std::cerr << "Exception reading from input file: " << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
        
    };
    
    
    
    
        int writeParamsToJsonFile(const std::string filePath) {
        
        
        try {
            
            nlohmann::json jsonParams = getJson();
            
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams.dump(4);  // Pretty print with 4 spaces
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
