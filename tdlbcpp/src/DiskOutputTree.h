
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
    
    GridParams grid;
    FlowParams<double> flow;
    ComputeUnitParams cuJson;
    
public:
    
    std::string rootDir;
    
    
    
    DiskOutputTree(std::string driveRoot, std::string _rootDir, GridParams grid, FlowParams<double> flow, ComputeUnitParams cuJson): grid(grid), flow(flow), cuJson(cuJson){
        
        rootDir = driveRoot + "/" + _rootDir;
        
        createDir(rootDir);
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
    
    
    
    std::string formatStep(tStep step){
        
        std::stringstream sstream;
        
        sstream << std::setw(8) << std::setfill('0') << patch::to_string(step);
        
        return sstream.str();
    }
    
    
    void createDir(std::string dir){
        
        std::cout << "Creating outputFilesTree " << dir << std::endl;
        mkdir(dir.c_str(), 0775);
        
    }
    
    //==================================================
    
    
    
    std::string formatDir(std::string prefix, std::string plot_type, tStep step) {
        return rootDir + "/" + prefix + "." + plot_type + ".V5.step_" + formatStep(step);
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
    
    
    std::string formatCaptureAtBladeAngleDir(tStep step, int angle, int blade_id, const std::string prefix="plot_"){
        
        return formatDir(prefix, "rotational_capture", step) + ".angle_" + patch::to_string(angle) + ".blade_id_" + patch::to_string(blade_id);
    }
    
    
    std::string formatAxisWhenBladeAngleDir(tStep step, int angle, const std::string prefix="plot"){
        
        return formatDir(prefix, "YZplane", step) + ".angle_" + patch::to_string(angle);
    }
    
    
    
    std::string formatRotatingSectorDir(tStep step, int angle, const std::string prefix="plot"){
        
        std::string ret = "";
        return ret;
    }
    
    
    
    
    std::string formatCUid(){
        return "CUid." + patch::to_string(cuJson.idi) + "." + patch::to_string(cuJson.idj) + "." + patch::to_string(cuJson.idk);
    }
    
    
    std::string formatQVecBinFileNamePath(std::string dir){
        return dir + "/" + formatCUid() + ".V5.QVec.bin";
    }
    
    std::string formatF3BinFileNamePath(std::string dir){
        return dir + "/" + formatCUid() + ".V5.F3.bin";
    }
    
    
    int writeBinFileJson(BinFileFormat format, RunningParams runParam){
        //Save Grid
        //save flow,
        //etc/
        format.writeParams(format.filePath + ".json");
    
        return 0;
        
    }
    
    
    
    
    
    
    
    
    
};


#endif


