//
//  Output.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include <cmath>


#include "ComputeUnit.h"



template <typename T, int size>
struct tDiskDense {
    T q[size];
};

template <typename T, int size>
struct tDiskGrid {
    uint16_t iGrid, jGrid, kGrid;
    T q[size];
};

template <typename T, int size>
struct tDiskColRow {
    uint16_t col, row;
    T q[size];
};

template <typename T, int size>
struct tDiskGridColRow {
    uint16_t iGrid, jGrid, kGrid;
    uint16_t col, row;
    T q[size];
};




