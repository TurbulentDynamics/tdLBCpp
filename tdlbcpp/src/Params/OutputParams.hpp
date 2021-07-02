
//
//  OutputData.hpp.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Niall OByrnes on 23/07/2018.
//  Copyright © 2018 Niall P. O'Byrnes. All rights reserved.
//

#ifndef Output_Data_hpp
#define Output_Data_hpp

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "json.h"




struct OrthoPlane {
    std::string name_root;
    tStep repeat = 0;
    tNi cutAt = 0;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float = 0;
    
    std::string QDataType;

    void getParamsFromJson(Json::Value jsonParams);
    Json::Value getJson();
};




struct Volume {
    std::string name_root;
    tStep repeat = 0;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;

    void getParamsFromJson(Json::Value jsonParams);
    Json::Value getJson();
};






struct Angle {
    std::string name_root;
    tStep repeat = 0;
    double degrees = 0;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;

    void getParamsFromJson(Json::Value jsonParams);
    Json::Value getJson();
};




struct PlaneAtAngle {
    std::string name_root;
    double degrees = 0;
    double tolerance = 0;
    tNi cutAt = 0;
    
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;

    void getParamsFromJson(Json::Value jsonParams);
    Json::Value getJson();
};



struct Sector {
    std::string name_root;
    tStep repeat = 0;
    
    
    double angle_infront_blade = 0.0f;
    double angle_behind_blade = 0.0f;
    
    int Q_output_len = 4;
    tStep start_at_step = 0;
    tStep end_at_repeat = 0;
    bool use_half_float;
    
    std::string QDataType;

    void getParamsFromJson(Json::Value jsonParams);
    Json::Value getJson();
};





struct OutputParams {
    
    
    std::string rootDir = "plot_root_dir";
    
    //    OutputConfig(){};
    std::vector<OrthoPlane> XY_planes;
    std::vector<OrthoPlane> XZ_planes;
    std::vector<OrthoPlane> YZ_planes;
    std::vector<Angle> capture_at_blade_angle;
    std::vector<PlaneAtAngle> YZ_plane_when_angle;
    std::vector<Volume> volumes;
    
    
    //Not Yet Implemenmted
    //std::vector<Sector> sectors;
    
    
    OutputParams(std::string rootDir):rootDir(rootDir){};
    
    
    void print_output_config_data();
    
    
    void add_XY_plane(std::string dir, tStep repeat, tNi cutAt,
                      int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        OrthoPlane p;
        p.name_root = dir;
        p.repeat = repeat;
        p.cutAt = cutAt;
        p.Q_output_len = Q_output_len;
        p.start_at_step = start_at_step;
        p.end_at_repeat = end_at_repeat;
        p.use_half_float = use_half_float;
        p.QDataType = QDataType;
        
        //        plane p = {dir, repeat, cutAt, Q_output_len, start_at_step, end_at_repeat, use_half_float, QDataType};
        XY_planes.push_back(p);
    }
    
    
    
    
    
    
    
    //formally axis
    void add_XZ_plane(const std::string dir, tStep repeat, tNi cutAt,
                      int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        OrthoPlane p;
        p.name_root = dir;
        p.repeat = repeat;
        p.cutAt = cutAt;
        p.Q_output_len = Q_output_len;
        p.start_at_step = start_at_step;
        p.end_at_repeat = end_at_repeat;
        p.use_half_float = use_half_float;
        p.QDataType = QDataType;
        
        //        plane p = {dir, repeat, cutAt, Q_output_len, start_at_step, end_at_repeat, use_half_float, QDataType};
        XZ_planes.push_back(p);
    }
    
    
    
    //Formaly slice
    void add_YZ_plane(const std::string dir, tStep repeat, tNi cutAt,
                      int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        OrthoPlane p;
        p.name_root = dir;
        p.repeat = repeat;
        p.cutAt = cutAt;
        p.Q_output_len = Q_output_len;
        p.start_at_step = start_at_step;
        p.end_at_repeat = end_at_repeat;
        p.use_half_float = use_half_float;
        p.QDataType = QDataType;
        
        YZ_planes.push_back(p);
    }
    
    
    
    
    
    
    
    void add_angle(const std::string dir,
                   tStep rotational_capture_repeat, double rotational_capture_behind_impeller_degrees,
                   int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_grid_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_grid_colrow_Q19";
        
        Angle a;
        a.name_root = dir;
        a.repeat = rotational_capture_repeat;
        a.degrees = rotational_capture_behind_impeller_degrees;
        a.Q_output_len = Q_output_len;
        a.start_at_step = start_at_step;
        a.end_at_repeat = end_at_repeat;
        a.use_half_float = use_half_float;
        a.QDataType = QDataType;
        
        capture_at_blade_angle.push_back(a);
        
    }
    
    
    void add_YZ_plane_at_angle(const std::string dir, double fixed_axis_capture_behind_impeller_degrees,
                               double tolerance, tNi cutAt, int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_colrow_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_colrow_Q19";
        
        PlaneAtAngle p;
        p.name_root = dir;
        p.degrees = fixed_axis_capture_behind_impeller_degrees;
        p.tolerance = tolerance;
        p.cutAt = cutAt;
        p.Q_output_len = Q_output_len;
        p.start_at_step = start_at_step;
        p.end_at_repeat = end_at_repeat;
        p.use_half_float = use_half_float;
        p.QDataType = QDataType;
        
        YZ_plane_when_angle.push_back(p);
        
    }
    
    
    
    
    void add_volume(const std::string dir, tStep plot_full_repeat,
                    int Q_output_len = 4, tStep start_at_step = 0, tStep end_at_repeat = 0, bool use_half_float=0){
        
        std::string QDataType = "tDisk_grid_Q4";
        if (Q_output_len == 19) QDataType = "tDisk_grid_Q19";
        
        
        Volume v;
        v.name_root = dir;
        v.repeat = plot_full_repeat;
        v.Q_output_len = Q_output_len;
        v.start_at_step = start_at_step;
        v.end_at_repeat = end_at_repeat;
        v.use_half_float = use_half_float;
        v.QDataType = QDataType;
        
        volumes.push_back(v);
        
    }

    template <typename ParamType> void getParamsFromJsonArray(Json::Value jsonArray, std::vector<ParamType> &array) {
       array.clear();
        for (Json::Value::ArrayIndex i = 0; i < jsonArray.size(); i++) {
            ParamType param;
            param.getParamsFromJson(jsonArray[i]);
            array.push_back(param);
        }
    }

    void getParamsFromJson(Json::Value jsonParams) {
        try
        {
            rootDir = jsonParams["rootDir"].asString();
            
            getParamsFromJsonArray(jsonParams["XY_planes"], XY_planes);
            getParamsFromJsonArray(jsonParams["XZ_planes"], XZ_planes);
            getParamsFromJsonArray(jsonParams["YZ_planes"], YZ_planes);
            getParamsFromJsonArray(jsonParams["capture_at_blade_angle"], capture_at_blade_angle);
            getParamsFromJsonArray(jsonParams["YZ_plane_when_angle"], YZ_plane_when_angle);
            getParamsFromJsonArray(jsonParams["volumes"], volumes);
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
        }
    }

    template <typename ParamType> Json::Value getJsonOfArray(std::vector<ParamType> &array) {
        Json::Value jsonArray = Json::arrayValue;
        for (ParamType param : array) {
            jsonArray.append(param.getJson());
        }
        return jsonArray;
    }
    
    Json::Value getJson() {
        try
        {
            Json::Value jsonParams;

            jsonParams["rootDir"] = rootDir;
    
            jsonParams["XY_planes"] = getJsonOfArray(XY_planes);
            jsonParams["XZ_planes"] = getJsonOfArray(XZ_planes);
            jsonParams["YZ_planes"] = getJsonOfArray(YZ_planes);
            jsonParams["capture_at_blade_angle"] = getJsonOfArray(capture_at_blade_angle);
            jsonParams["YZ_plane_when_angle"] = getJsonOfArray(YZ_plane_when_angle);
            jsonParams["volumes"] = getJsonOfArray(volumes);

            return jsonParams;
        }
        catch (std::exception &e)
        {

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
        
    
    
};

void OrthoPlane::getParamsFromJson(Json::Value jsonParams)    
{
    try
    {
        name_root = jsonParams["name_root"].asString();
        repeat = jsonParams["repeat"].asUInt64();
        cutAt = jsonParams["cutAt"].asInt64();

        Q_output_len = jsonParams["Q_output_len"].asInt();
        start_at_step = jsonParams["start_at_step"].asUInt64();
        end_at_repeat = jsonParams["end_at_repeat"].asUInt64();
        use_half_float = jsonParams["use_half_float"].asBool();

        QDataType = jsonParams["QDataType"].asString();

    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached parsing arguments: "
        << e.what() << ", application will now exit" << std::endl;
    }
}

Json::Value OrthoPlane::getJson()
{
    try
    {
        Json::Value jsonParams;

        jsonParams["name_root"] = name_root;
        jsonParams["repeat"] = (tStep)repeat;
        jsonParams["cutAt"] = (tNi)cutAt;

        jsonParams["Q_output_len"] = Q_output_len;
        jsonParams["start_at_step"] = (tStep)start_at_step;
        jsonParams["end_at_repeat"] = (tStep)end_at_repeat;
        jsonParams["use_half_float"] = use_half_float;

        jsonParams["QDataType"] = QDataType;

        return jsonParams;
    }
    catch (std::exception &e)
    {
        std::cerr << "Unhandled Exception reached parsing arguments: "
                    << e.what() << ", application will now exit" << std::endl;
        return "";
    }
}

void Volume::getParamsFromJson(Json::Value jsonParams)
{
    try
    {
        name_root = jsonParams["name_root"].asString();
        repeat = jsonParams["repeat"].asUInt64();

        Q_output_len = jsonParams["Q_output_len"].asInt();
        start_at_step = jsonParams["start_at_step"].asUInt64();
        end_at_repeat = jsonParams["end_at_repeat"].asUInt64();
        use_half_float = jsonParams["use_half_float"].asBool();

        QDataType = jsonParams["QDataType"].asString();

    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached parsing arguments: "
        << e.what() << ", application will now exit" << std::endl;
    }
}

Json::Value Volume::getJson()
{
    try
    {
        Json::Value jsonParams;

        jsonParams["name_root"] = name_root;
        jsonParams["repeat"] = (tStep)repeat;

        jsonParams["Q_output_len"] = Q_output_len;
        jsonParams["start_at_step"] = (tStep)start_at_step;
        jsonParams["end_at_repeat"] = (tStep)end_at_repeat;
        jsonParams["use_half_float"] = use_half_float;

        jsonParams["QDataType"] = QDataType;

        return jsonParams;        

    }
    catch (std::exception &e)
    {
        std::cerr << "Unhandled Exception reached parsing arguments: "
                    << e.what() << ", application will now exit" << std::endl;
        return "";
    }
}

void Angle::getParamsFromJson(Json::Value jsonParams)
{
    try
    {
        name_root = jsonParams["name_root"].asString();
        repeat = jsonParams["repeat"].asUInt64();
        degrees = jsonParams["degrees"].asDouble();

        Q_output_len = jsonParams["Q_output_len"].asInt();
        start_at_step = jsonParams["start_at_step"].asUInt64();
        end_at_repeat = jsonParams["end_at_repeat"].asUInt64();
        use_half_float = jsonParams["use_half_float"].asBool();

        QDataType = jsonParams["QDataType"].asString();

    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached parsing arguments: "
        << e.what() << ", application will now exit" << std::endl;
    }
}

Json::Value Angle::getJson()
{
    try
    {
        Json::Value jsonParams;

        jsonParams["name_root"] = name_root;
        jsonParams["repeat"] = (tStep)repeat;
        jsonParams["degrees"] = (double)degrees;

        jsonParams["Q_output_len"] = Q_output_len;
        jsonParams["start_at_step"] = (tStep)start_at_step;
        jsonParams["end_at_repeat"] = (tStep)end_at_repeat;
        jsonParams["use_half_float"] = use_half_float;

        jsonParams["QDataType"] = QDataType;

        return jsonParams;
    }
    catch (std::exception &e)
    {
        std::cerr << "Unhandled Exception reached parsing arguments: "
                    << e.what() << ", application will now exit" << std::endl;
        return "";
    }
}

void PlaneAtAngle::getParamsFromJson(Json::Value jsonParams)
{
    try
    {
        name_root = jsonParams["name_root"].asString();
        degrees = jsonParams["degrees"].asDouble();
        tolerance = jsonParams["tolerance"].asDouble();
        cutAt = jsonParams["cutAt"].asInt64();


        Q_output_len = jsonParams["Q_output_len"].asInt();
        start_at_step = jsonParams["start_at_step"].asUInt64();
        end_at_repeat = jsonParams["end_at_repeat"].asUInt64();
        use_half_float = jsonParams["use_half_float"].asBool();

        QDataType = jsonParams["QDataType"].asString();

    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached parsing arguments: "
        << e.what() << ", application will now exit" << std::endl;
    }
}

Json::Value PlaneAtAngle::getJson()
{
    try
    {
        Json::Value jsonParams;

        jsonParams["name_root"] = name_root;
        jsonParams["degrees"] = (double)degrees;
        jsonParams["tolerance"] = (double)tolerance;
        jsonParams["cutAt"] = (tNi)cutAt;


        jsonParams["Q_output_len"] = Q_output_len;
        jsonParams["start_at_step"] = (tStep)start_at_step;
        jsonParams["end_at_repeat"] = (tStep)end_at_repeat;
        jsonParams["use_half_float"] = use_half_float;

        jsonParams["QDataType"] = QDataType;


        return jsonParams;
    }
    catch (std::exception &e)
    {
        std::cerr << "Unhandled Exception reached parsing arguments: "
                    << e.what() << ", application will now exit" << std::endl;
        return "";
    }
}

void Sector::getParamsFromJson(Json::Value jsonParams)
{
    try
    {
        name_root = jsonParams["name_root"].asString();
        repeat = jsonParams["repeat"].asUInt64();


        angle_infront_blade = jsonParams["angle_infront_blade"].asDouble();
        angle_behind_blade = jsonParams["angle_behind_blade"].asDouble();

        Q_output_len = jsonParams["Q_output_len"].asInt();
        start_at_step = jsonParams["start_at_step"].asUInt64();
        end_at_repeat = jsonParams["end_at_repeat"].asUInt64();
        use_half_float = jsonParams["use_half_float"].asBool();

        QDataType = jsonParams["QDataType"].asString();

    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached parsing arguments: "
        << e.what() << ", application will now exit" << std::endl;
    }
}

Json::Value Sector::getJson()
{
    try
    {
        Json::Value jsonParams;

        jsonParams["name_root"] = name_root;
        jsonParams["repeat"] = (tStep)repeat;


        jsonParams["angle_infront_blade"] = (double)angle_infront_blade;
        jsonParams["angle_behind_blade"] = (double)angle_behind_blade;

        jsonParams["Q_output_len"] = Q_output_len;
        jsonParams["start_at_step"] = (tStep)start_at_step;
        jsonParams["end_at_repeat"] = (tStep)end_at_repeat;
        jsonParams["use_half_float"] = use_half_float;

        jsonParams["QDataType"] = QDataType;


        return jsonParams;
    }
    catch (std::exception &e)
    {
        std::cerr << "Unhandled Exception reached parsing arguments: "
                    << e.what() << ", application will now exit" << std::endl;
        return "";
    }
}

#endif
