
//
//  DiskOutputTreeV5.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Niall OByrnes on 23/07/2018.
//  Copyright © 2018 Niall P. O'Byrnes. All rights reserved.
//

#ifndef DiskOutputTreeV5_hpp
#define DiskOutputTreeV5_hpp

#include <sys/stat.h> // mkdir
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <cstdint>
#include <map>
#include <string>
#include <time.h>
#include "Params/json.h"


#include "Header.h"

#include "Params/Grid.hpp"
#include "Params/ComputeUnitParams.hpp"
#include "Params/Flow.hpp"





//https://stackoverflow.com/questions/12975341/to-string-is-not-a-member-of-std-says-g-mingw
#include <sstream>
namespace patch
{
template < typename T > std::string to_string( const T& n )
{
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
}
}






class DiskOutputTree {
    
private:

    ComputeUnitParams cu;
    
    Json::Value cuJson;

    Json::Value grid;
    Json::Value flow;
    Json::Value running;
    Json::Value output;
    Json::Value checkpoint;

    
public:
    
    std::string outputRootDir = "output_dir";
    std::string checkpointRootDir = "checkpoint_dir";

    
    DiskOutputTree(CheckpointParams checkpoint, OutputParams output){
        outputRootDir = output.rootDir;
        createDir(outputRootDir);

        checkpointRootDir = checkpoint.checkpoint_root_dir;
        createDir(checkpointRootDir);
    };

    
    DiskOutputTree(ComputeUnitParams cu, Json::Value grid, Json::Value flow, Json::Value running, Json::Value output, Json::Value checkpoint): cu(cu),  grid(grid), flow(flow), running(running), output(output), checkpoint(checkpoint) {
        
        cuJson = cu.getJson();
        
        outputRootDir = output["root_dir"].asString();
        createDir(outputRootDir);

        checkpointRootDir = checkpoint["checkpoint_root_dir"].asString();
        createDir(checkpointRootDir);
    };
    
    
    DiskOutputTree(ComputeUnitParams cu1, GridParams grid1, FlowParams<double> flow1, RunningParams running1,  OutputParams output1, CheckpointParams checkpoint1){
        
        cu = cu1;
        
        cuJson = cu1.getJson();

        grid = grid1.getJson();
        flow =  flow1.getJson();
        running = running1.getJson();
        output = output1.getJson();
        checkpoint = checkpoint1.getJson();

        
        outputRootDir = output1.rootDir;
        createDir(outputRootDir);

        checkpointRootDir = checkpoint1.checkpoint_root_dir;
        createDir(checkpointRootDir);
    };
    

    void setParams(ComputeUnitParams cu1, GridParams grid1, FlowParams<double> flow1, RunningParams running1, OutputParams output1, CheckpointParams checkpoint1){
        
        cu = cu1;
        cuJson = cu1.getJson();
        
        grid = grid1.getJson();
        flow =  flow1.getJson();
        running = running1.getJson();
        output = output1.getJson();
        checkpoint = checkpoint1.getJson();
        
    };
        
    

    

    
    

    
    inline bool pathExists(std::string path) {
        
        if (FILE *file = fopen(path.c_str(), "r")) {
            fclose(file);
            return true;
        } else {
            return false;
        }
    }
    
    
    inline bool fileExists(std::string path) {
        return pathExists(path);
    }

    
    void createDir(std::string dir){
        
        std::cout << "DiskOutputTree creating directory: " << dir << std::endl;
        mkdir(dir.c_str(), 0775);
        
    }
    
    
    //==================================================
    
    
    //https://stackoverflow.com/questions/16357999/current-date-and-time-as-string
    std::string getTimeNowAsString(){

        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];

        time (&rawtime);
        timeinfo = localtime(&rawtime);

        strftime(buffer, sizeof(buffer), "%Y_%m_%d_%H_%M_%S", timeinfo);
        std::string time_now(buffer);

        //https://www.techiedelight.com/replace-occurrences-character-string-cpp/
        size_t pos;
        while ((pos = time_now.find("-")) != std::string::npos) {time_now.replace(pos, 1, "_");}
        while ((pos = time_now.find(" ")) != std::string::npos) {time_now.replace(pos, 1, "_");}
        while ((pos = time_now.find(":")) != std::string::npos) {time_now.replace(pos, 1, "_");}


        return time_now;
    }


    void setRunDir(std::string runDir1){

        outputRootDir = runDir1;
    }
    
    std::string getRunDirWithTimeAndParams(std::string prefix, tNi gridX, int re_m, bool les, float uav, tStep step = 0){

        std::string str = prefix + "_";

        if (step) str += "step_" + std::to_string(step) + "__";

        str += "datetime_" + getTimeNowAsString() + "_";
        str += "gridx_" + std::to_string(gridX) + "_";
        str += "re_" + std::to_string(re_m) + "_";
        str += "les_" + std::to_string(les) + "_";
        str += "uav_" + std::to_string(uav);

        return str;
    }
    
    
    //==================================================
    
    
    
    std::string formatStep(tStep step){
        
        std::stringstream sstream;
        
        sstream << std::setw(8) << std::setfill('0') << patch::to_string(step);
        
        return sstream.str();
    }
    
    
    std::string formatDir(std::string prefix, std::string plotType, tStep step) {
        return outputRootDir + "/" + prefix + "." + plotType + ".V5.step_" + formatStep(step);
    }
    
    std::string formatXYPlaneDir(tStep step, tNi atK, const std::string prefix="plot"){
        
        return formatDir(prefix, "XYplane", step) + ".cut_" + patch::to_string(atK);
    }
    
    //Formally Axis
    std::string formatXZPlaneDir(tStep step, tNi atJ, const std::string prefix="plot"){
        
        //TOFIX TODO
        
        std::string jijm =formatDir(prefix, "XZplane", step) + ".cut_" + patch::to_string(atJ);
        mkdir(jijm.c_str(), 0775);
        
        return jijm;
    }
    
    //Formally Slice
    std::string formatYZPlaneDir(tStep step, tNi atI, const std::string prefix="plot"){
        
        return formatDir(prefix, "YZplane", step) + ".cut_" + patch::to_string(atI);
    }
    
    std::string formatVolumeDir(tStep step, const std::string prefix="plot"){
        return formatDir(prefix, "volume", step);
    }
    
    
    std::string formatCaptureAtBladeAngleDir(tStep step, int angle, int bladeId, const std::string prefix="plot"){
        
        return formatDir(prefix, "rotational_capture", step) + ".angle_" + patch::to_string(angle) + ".bladeId_" + patch::to_string(bladeId);
    }
    
    
    std::string formatAxisWhenBladeAngleDir(tStep step, int angle, const std::string prefix="plot"){
        
        return formatDir(prefix, "YZplane", step) + ".angle_" + patch::to_string(angle);
    }
    
    
    
    std::string formatRotatingSectorDir(tStep step, int angle, const std::string prefix="plot"){
        
        std::string ret = "";
        return ret;
    }
    
    
    
    
    std::string formatCUid(){
        return "CUid." + patch::to_string(cu.idi) + "." + patch::to_string(cu.idj) + "." + patch::to_string(cu.idk);
    }
    
    
    std::string formatQVecBinFileNamePath(std::string dir){
        return dir + "/" + formatCUid() + ".V5.QVec.bin";
    }
    
    std::string formatF3BinFileNamePath(std::string dir){
        return dir + "/" + formatCUid() + ".V5.F3.bin";
    }
    
    
    int writeAllParamsJson(BinFileParams format, RunningParams runParam){

        try {

            Json::Value jsonParams;

            jsonParams["ComputeUnitParams"] = cuJson;

            jsonParams["BinFileParams"] = format.getJson();
            jsonParams["GridParams"] = grid;
            jsonParams["FlowParams"] = flow;
            jsonParams["RunningParams"] = running;
            jsonParams["OutputParams"] = output;
            jsonParams["CheckpointParams"] = checkpoint;

            
            std::string path = format.filePath + ".json";
            
            std::ofstream out(path.c_str(), std::ofstream::out);
            out << jsonParams;
            out.close();

            return 0;
        }
        catch(std::exception& e) {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return 1;
        }

    
        return 0;
        
    }
    
    
    int readAllParamsJson(std::string filePath, BinFileParams format, RunningParams runParam){


        try {
            std::ifstream in(filePath.c_str());
            Json::Value jsonParams;
            in >> jsonParams;
            in.close();

            
            //TODO


            cuJson = jsonParams["ComputeUnitParams"];
            cu.getParamsFromJson(cuJson);
            
            //Not necessary atm
//            binFileJson = jsonParams["BinFileParams"];

            grid = jsonParams["GridParams"];
            flow = jsonParams["FlowParams"];
            running = jsonParams["RunningParams"];
            output = jsonParams["OutputParams"];
            checkpoint = jsonParams["CheckpointParams"];



        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
        }
    

    
        return 0;
        
    }
    
    
    //===================================
    
    
    
    std::string getCheckpointDirName(RunningParams run, bool create=true){
       
        
        std::string dirName = checkpoint["checkpoint_root_dir"].asString() + "/step_" + std::to_string(run.step);
     
        
//        std::string dirName = checkpointRootDir + "/step_" + std::to_string(run.step);
     
        if (create) {
            createDir(dirName);
        }

        return dirName;
    }
    
    std::string getCheckpointFilePath(std::string dirName, std::string unit_name, std::string matrix){

        
        std::string path = dirName + "/checkpoint_grid." + std::to_string(cu.idi) + "." + std::to_string(cu.idj) + "." + std::to_string(cu.idk) + ".";

        path += unit_name + "." + matrix;

        return path;
    }
    


    
    
    
};


#endif


