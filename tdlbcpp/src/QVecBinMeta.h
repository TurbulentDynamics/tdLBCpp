//
//  define_datastructures.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Nile Ó Broin on 08/01/2019.
//  Copyright © 2019 Nile Ó Broin. All rights reserved.
//

#ifndef QvecBinFile_Dims_hpp
#define QvecBinFile_Dims_hpp

#include <stdlib.h>

#include "json.h"

#include "Header.h"
#include "BaseParams.h"


struct QVecBinMeta {

    std::string name = "QVecBinMeta";
    tNi grid_x = 0;
    tNi grid_y = 0;
    tNi grid_z = 0;


    int ngx = 0;
    int ngy = 0;
    int ngz = 0;

    int idi = 0;
    int idj = 0;
    int idk = 0;


    std::string struct_name = "";
    unsigned long int bin_file_size_in_structs = 0;



    std::string coords_type = "";
    bool has_grid_coords;
    bool has_col_row_coords;



    std::string Q_data_type = "";
    int Q_output_length = 0;



    void set_dims(int ngx, int ngy, int ngz, tNi snx, tNi sny, tNi snz){

        ngx = ngx;
        ngy = ngy;
        ngz = ngz;

        grid_x = snx;
        grid_y = sny;
        grid_z = snz;
    }

    void set_ids(int idi, int idj, int idk){

        idi = idi;
        idj = idj;
        idk = idk;

    }

    void set_file_content(std::string struct_name, unsigned long int bin_file_size_in_structs,
                          std::string coords_type, bool has_grid_coords, bool has_col_row_coords,
                          std::string Q_data_type, int Q_output_length)
    {

        struct_name = struct_name;
        bin_file_size_in_structs = bin_file_size_in_structs;

        coords_type = coords_type;
        has_grid_coords = has_grid_coords;
        has_col_row_coords = has_col_row_coords;

        Q_data_type = Q_data_type;
        Q_output_length = Q_output_length;


    }



    

    std::string get_json_filepath_from_Qvec_filepath(const std::string Qvec_filepath){
        return Qvec_filepath + ".json";
    }

    bool Qvec_json_file_exists(const std::string Qvec_filepath){
        std::string json_filepath = get_json_filepath_from_Qvec_filepath(Qvec_filepath);
        return file_exists(json_filepath);
    }



    
    QVecBinMeta get_from_json_filepath(const std::string filepath){


        QVecBinMeta d;

        try
        {
            std::ifstream in(filepath.c_str());
            Json::Value dim_json;
            in >> dim_json;


            d.grid_x = (tNi)dim_json["grid_x"].asInt();
            d.grid_y = (tNi)dim_json["grid_y"].asInt();
            d.grid_z = (tNi)dim_json["grid_z"].asInt();

            d.ngx = (int)dim_json["ngx"].asInt();
            d.ngy = (int)dim_json["ngy"].asInt();
            d.ngz = (int)dim_json["ngz"].asInt();

            d.idi = (int)dim_json["idi"].asInt();
            d.idj = (int)dim_json["idj"].asInt();
            d.idk = (int)dim_json["idk"].asInt();


            d.struct_name = dim_json["struct_name"].asString();
            d.bin_file_size_in_structs = dim_json["bin_file_size_in_structs"].asUInt64();


            d.coords_type = dim_json["coords_type"].asString();
            d.has_grid_coords = dim_json["has_grid_coords"].asBool();
            d.has_col_row_coords = dim_json["has_col_row_coords"].asBool();


            d.Q_data_type = dim_json["Q_data_type"].asString();
            d.Q_output_length = (int)dim_json["Q_output_length"].asInt();


            in.close();
            return d;
        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return d;
        }
    }





    //template <class Tpos, class Tdata>
    int save_json_to_Qvec_filepath(const std::string qvec_filepath){

        std::string json_filepath = get_json_filepath_from_Qvec_filepath(qvec_filepath);

        try
        {
            Json::Value dim_json;

            dim_json["name"] = name;

            dim_json["grid_x"] = (int)grid_x;
            dim_json["grid_y"] = (int)grid_y;
            dim_json["grid_z"] = (int)grid_z;


            dim_json["ngx"] = (int)ngx;
            dim_json["ngy"] = (int)ngy;
            dim_json["ngz"] = (int)ngz;

            dim_json["idi"] = (int)idi;
            dim_json["idj"] = (int)idj;
            dim_json["idk"] = (int)idk;


            dim_json["struct_name"] = struct_name;
            dim_json["bin_file_size_in_structs"] = (uint64_t)bin_file_size_in_structs;


            dim_json["coords_type"] = coords_type;
            dim_json["has_grid_coords"] = has_grid_coords;
            dim_json["has_col_row_coords"] = has_col_row_coords;


            dim_json["Q_data_type"] = Q_data_type;
            dim_json["Q_output_length"] = (int)Q_output_length;



            std::ofstream out(json_filepath.c_str(), std::ofstream::out);
            out << dim_json;
            out.close();

            return 0;
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
        std::cout << std::endl << "QvecBinFile_Dims "
        << " bin_file_size_in_structs:" << bin_file_size_in_structs

        << " grid_x:" << grid_x
        << " grid_y:" << grid_y
        << " grid_z:" << grid_z


        << " ngx:" << ngx
        << " ngx:" << ngy
        << " ngx:" << ngz


        << " idi:" << idi
        << " idj:" << idj
        << " idk:" << idk

        << " struct_name:" << struct_name
        << " bin_file_size_in_structs:" << bin_file_size_in_structs


        << " coords_type:" << coords_type
        << " has_grid_coords:" << has_grid_coords
        << " has_col_row_coords:" << has_col_row_coords


        << " Q_data_type:" << Q_data_type
        << " Q_output_length:" << Q_output_length


        << std::endl
        << std::endl;
    }

};






#endif   /*  Dims_hpp  */
