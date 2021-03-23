
//
//  Output_Qvec.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Niall OByrnes on 23/07/2018.
//  Copyright © 2018 Niall P. O'Byrnes. All rights reserved.
//

#ifndef PlotDirV4_hpp
#define PlotDirV4_hpp

#include <sys/stat.h> // mkdir
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <cstdint>
#include <map>


#include "Header.h"
#include "BaseParams.h"

#include "GridParams.hpp"



//#if IS_INTEL_CXX == 1
//#include "half.hpp"
//#endif



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


enum PlotDirKind {
    XYplane,
    XZplane,
    YZplane,
    rotational,
    volume,
    //    case sector = "sector"
};

static std::map< PlotDirKind, const char * > PlotDirKindMap = {
    {XYplane, "XYplane"},
    {XZplane, "XZplane"},
    {YZplane, "YZplane"},
    {rotational, "rotational_capture"},
    {volume, "volume"}
};


enum binExt {bin, binJson, rho, ux, uy, uz, vorticity};

static std::map< binExt, const char * > binExtMap = {
    {bin, "bin"},
    {binJson, "bin.json"},
    {rho, "rho.bin"},
    {ux, "ux.bin"},
    {uy, "uy.bin"},
    {uz, "uz.bin"},
    {vorticity, "vorticity.bin"}
};


enum QvecNames {Qvec, F};

static std::map< QvecNames, const char * > QvecNamesMap = {
    {Qvec, "Qvec"},
    {F, "Qvec.F"}
};



struct PlotDir {

    std::string path;
    int idi, idj, idk;

    
    PlotDir(std::string path, int idi, int idj, int idk):path(path), idi(idi), idj(idj), idk(idk){
    };

    PlotDir(std::string path, tNi idi, tNi idj, tNi idk):path(path), idi((int)idi), idj((int)idj), idk((int)idk){
    };
    
    std::string get_Qvec_fileroot(QvecNames nameEnum, int _idi, int _idj, int _idk) {
                
        return path + "/" + QvecNamesMap[nameEnum] + ".node." + std::to_string(_idi) + "." + std::to_string(_idj) + "." + std::to_string(_idk) + ".V4";
    }
    
    std::string get_Qvec_filename(QvecNames name, int _idi, int _idj, int _idk) {
        return get_Qvec_fileroot(name, _idi, _idj, _idk) + ".bin";
    }
    
    std::string get_my_Qvec_filename(QvecNames name) {
        return get_Qvec_filename(name, idi, idj, idk);
    }
    
    std::string get_node000_Qvec_filename(QvecNames name) {
        return get_Qvec_filename(name, 0, 0, 0);
    }
    
    std::string get_node000_Qvec_filename_json(QvecNames name) {
        return get_Qvec_filename(name, 0, 0, 0) + ".json";
    }
};


class OutputDir {
    
private:
        
public:
    
    GridParams grid;
    std::string rootDir;
    
    OutputDir(){};
    
    
    OutputDir(std::string rootDir, GridParams grid):rootDir(rootDir), grid(grid){
    };
    

    
    std::string format_step(tStep step){
        
        std::stringstream sstream;
        
        sstream << std::setw(8) << std::setfill('0') << patch::to_string(step);
        
        return sstream.str();
    }
    
    
    

    
    std::string get_start(std::string name, std::string plot_type, int Q_length, tStep step) {
        return rootDir + "/" + name + "." + plot_type + ".V_4.Q_" + patch::to_string(Q_length) + ".step_" + format_step(step);
    }
    
    void createDir(std::string dir){

        std::cout << dir << std::endl;
        mkdir(dir.c_str(), 0775);
    
    }

                       
    
    std::string get_XY_plane_dir(tStep step, tNi at_k, int Q_length, const std::string name="plot"){

        return get_start(name, "XYplane", Q_length, step) + ".cut_" + patch::to_string(at_k);
    }
    
        
        
    //Formally Axis
    std::string get_XZ_plane_dir(tStep step, tNi at_j, int Q_length, const std::string name="plot"){
        
        return get_start(name, "XZplane", Q_length, step) + ".cut_" + patch::to_string(at_j);
    }
    
    //Formally Slice
    std::string get_YZ_plane_dir(tStep step, tNi at_i, int Q_length, const std::string name="plot"){
        
        return get_start(name, "YZplane", Q_length, step) + ".cut_" + patch::to_string(at_i);
    }
    
    
    
    std::string get_volume_dir(tStep step, int Q_length, const std::string name="plot"){
        return get_start(name, "volume", Q_length, step);
    }
    
    
    
    
    
    
    std::string get_capture_at_blade_angle_dir(tStep step, int angle, int blade_id, int Q_length, const std::string name="plot_"){
        
        return get_start(name, "rotational_capture", Q_length, step) + ".angle_" + patch::to_string(angle) + ".blade_id_" + patch::to_string(blade_id);
    }
    
    
    std::string get_axis_when_blade_angle_dir(tStep step, int angle, int Q_length, const std::string name="plot"){
        
        return get_start(name, "YZplane", Q_length, step) + ".angle_" + patch::to_string(angle);
    }
    
    
    
    std::string get_rotating_sector_dir(tStep step, int angle, int Q_length, const std::string name="plot"){
        
        std::string ret = "";
        return ret;
    }
    
    
    
    std::string get_dir_delta(int delta, const std::basic_string<char> &load_dir) {
        using namespace std;
        
        
        if (!plot_type_in_dir("cut", load_dir)) return "not_found";
        
        
        string delta_load_dir = load_dir;
        
        size_t index = load_dir.find("cut_");
        
        int cut = stoi(load_dir.substr (index + 4));
        
        string delta_cut = "cut_" + patch::to_string(cut + delta);
        
        delta_load_dir.replace(index, string::npos, delta_cut);
        
        //    std::cout << "String" << load_dir << " Delta String   " << delta_load_dir << std::endl;
        
        return delta_load_dir;
    }
    
    
    
    
    bool plot_type_in_dir(const std::basic_string<char> &type, const std::basic_string<char> &load_dir) {
        using namespace std;
        
        size_t found = load_dir.find(type);
        if (found != string::npos) {
            //        cout << "found  " << type << " in " << load_dir << endl;
            return true;
        }
        return false;
    }
    
    
    std::string get_plot_type_from_directory(const std::basic_string<char> &load_dir) {
        
        if (plot_type_in_dir("XYplane", load_dir)) return "XYplane";
        else if (plot_type_in_dir("XZplane", load_dir)) return "XZplane";
        else if (plot_type_in_dir("YZplane", load_dir)) return "YZplane";
        
        else if (plot_type_in_dir("volume", load_dir)) return "volume";
        
        else if (plot_type_in_dir("rotational_capture", load_dir)) return "rotational_capture";
        
        
        
        return "";
    }
    
    
    
    
    
};








#endif














