//
//  BaseParams.h
//  tdLB_with_Soklev
//
//  Created by Niall Ó Broin on 20/12/2020.
//  Copyright © 2019 Nile Ó Broin. All rights reserved.
//

#pragma once


#include <stdlib.h>
#include <iostream>
#include <sys/stat.h> // mkdir

#include "Header.h"


inline std::string get_base_filepath(std::string name, tNi idi, tNi idj, tNi idk, const std::string dir="."){
    return dir + "/" + name + "_Dim." + std::to_string(idi) + "." + std::to_string(idj) + "." + std::to_string(idk);
}


inline std::string get_filepath(std::string name, tNi idi, tNi idj, tNi idk, const std::string dir="."){
    return get_base_filepath(name, idi, idj, idk, dir) + ".V4.json";
}


inline std::string get_bin_filepath_v3(std::string name, tNi idi, tNi idj, tNi idk, const std::string dir="."){
    return get_base_filepath(name, idi, idj, idk, dir) + ".V3.bin";
}




inline bool path_exists(std::string path) {

    if (FILE *file = fopen(path.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}


inline bool file_exists(std::string path) {
    return path_exists(path);
}











