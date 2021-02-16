//
//  define_datastructures.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Nile Ó Broin on 08/01/2019.
//  Copyright © 2019 Nile Ó Broin. All rights reserved.
//

#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "json.h"

#include "BaseParams.hpp"



//Grid Params are the maximum extent of a regular lattice.
struct GridParams
{
    tNi ngx = 0;
    tNi ngy = 0;
    tNi ngz = 0;

    tNi x = 0;
    tNi y = 0;
    tNi z = 0;
    
    


    GridParams get_from_json_filepath(const std::string filepath){

        GridParams g;

        try
        {
            std::ifstream in(filepath.c_str());
            Json::Value dim_json;
            in >> dim_json;

            g.ngx = (tNi)dim_json["ngx"].asInt();
            g.ngy = (tNi)dim_json["ngy"].asInt();
            g.ngz = (tNi)dim_json["ngz"].asInt();

            g.x = (tNi)dim_json["x"].asInt();
            g.y = (tNi)dim_json["y"].asInt();
            g.z = (tNi)dim_json["z"].asInt();

            in.close();


        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return g;
        }

        return g;

    };




    int save_json_to_filepath(const std::string filepath){
        try
        {

            Json::Value dim_json;

            dim_json["name"] = "GridParams";

            dim_json["ngx"] = (int)ngx;
            dim_json["ngy"] = (int)ngy;
            dim_json["ngz"] = (int)ngz;

            dim_json["x"] = (int)x;
            dim_json["y"] = (int)y;
            dim_json["z"] = (int)z;

            std::ofstream out(filepath.c_str(), std::ofstream::out);
            out << dim_json;
            out.close();


        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return 1;
        }

        return 0;
    }




    void print(){

        std::cout
        << " name:" << "GridParams"
        << " ngx:" << ngx
        << " ngx:" << ngy
        << " ngx:" << ngz
        << " x:" << x
        << " y:" << y
        << " z:" << y
        << std::endl;

    }

};//end of struct
