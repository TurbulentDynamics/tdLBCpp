
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

    
    Json::Value grid;
    Json::Value flow;
    Json::Value running;
    Json::Value checkpoint;
    Json::Value cuJson;
    Json::Value output;
    
    
public:
    
    std::string rootDir = ".";
    std::string runDir = "output_debug";

    
    DiskOutputTree(std::string _rootDir){
        rootDir = _rootDir;
        createDir(rootDir);
    };
    
    DiskOutputTree(std::string _diskDir, std::string _rootDir){
        rootDir = _diskDir + "/" + _rootDir;
        createDir(rootDir);
    };
    

    
    DiskOutputTree(std::string diskDir, std::string _rootDir, ComputeUnitParams cu, Json::Value grid, Json::Value flow, Json::Value running, Json::Value checkpoint, Json::Value cuJson, Json::Value output): cu(cu), grid(grid), flow(flow), running(running), checkpoint(checkpoint), cuJson(cuJson), output(output){
        
        
        rootDir = diskDir + "/" + _rootDir;
        
        createDir(rootDir);
    };
    
    
    
    DiskOutputTree(std::string diskDir, std::string _rootDir, ComputeUnitParams cu1, GridParams grid1, FlowParams<double> flow1, RunningParams running1, CheckpointParams checkpoint1, OutputParams output1){
        
        cu = cu1;
        
        grid = grid1.getJson();
        flow =  flow1.getJson();
        running = running1.getJson();
        checkpoint = checkpoint1.getJson();
        cuJson = cu1.getJson();
        output = output1.getJson();

        
        rootDir = diskDir + "/" + _rootDir;
        
        createDir(rootDir);
    };
    

    void setParams(ComputeUnitParams cu1, GridParams grid1, FlowParams<double> flow1, RunningParams running1, CheckpointParams checkpoint1, OutputParams output1){
        
        cu = cu1;
        
        grid = grid1.getJson();
        flow =  flow1.getJson();
        running = running1.getJson();
        checkpoint = checkpoint1.getJson();
        cuJson = cu1.getJson();
        output = output1.getJson();

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
        
        std::cout << "Creating DiskOutputTree " << dir << std::endl;
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


    void setRunDirWithTimeAndParams(std::string prefix, tNi gridX, int re_m, bool les, float uav, tStep step = 0){

        std::string str = rootDir + "/" + prefix + "_";

        if (step) str += "step_" + std::to_string(step) + "__";

        str += "datetime_" + getTimeNowAsString() + "_";
        str += "gridx_" + std::to_string(gridX) + "_";
        str += "re_" + std::to_string(re_m) + "_";
        str += "les_" + std::to_string(les) + "_";
        str += "uav_" + std::to_string(uav);

        runDir = str;
        createDir(str);
    }
    
    
    //==================================================
    
    
    
    std::string formatStep(tStep step){
        
        std::stringstream sstream;
        
        sstream << std::setw(8) << std::setfill('0') << patch::to_string(step);
        
        return sstream.str();
    }
    
    
    std::string formatDir(std::string prefix, std::string plotType, tStep step) {
        return runDir + "/" + prefix + "." + plotType + ".V5.step_" + formatStep(step);
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
    
    
    int writeBinFileJson(BinFileParams format, RunningParams runParam){

        //TODO Write to file format.filePath + ".json"
        //{filePath: "path",
        //structName" : ""
//        etc
//    grid: { ngx:ngx, }
//    flow: {flow:uav...

//        for all
//        Json::Value running;
//        Json::Value checkpoint;
//        Json::Value cuJson;
//        Json::Value output;

        
        
//        writeParams(format.filePath + ".json");
    
        return 0;
        
    }
    
    
    
    
    
    
    
    
    
};


#endif


