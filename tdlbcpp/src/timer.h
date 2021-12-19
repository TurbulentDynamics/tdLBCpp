//
//  timer.hpp
//  stirred-tank-3d-xcode-cpp
//
//  Created by Niall OByrnes on 23/07/2018.
//  Copyright © 2018 Niall P. O'Byrnes. All rights reserved.
//

#ifndef timer_h
#define timer_h

#include <stdio.h>
#include <iostream>
#include <iomanip>

#include "Header.h"


#define MAX_FUNC 100


double get_wall_time();


class Multi_Timer {

private:

    int rank;

    double epoch = 0.0;
    double get_now();

    bool timer_files_created = 0;

    int block[MAX_FUNC];
    int stream[MAX_FUNC];



    double start_time[MAX_FUNC];
    double elapsed[MAX_FUNC];
    double average[MAX_FUNC];

    std::string getAveragePerFunction(tStep step, int bloc, int stream, std::string name, double start_time, double elapsed);

public:

    Multi_Timer(int rank):rank(rank){
        reset_everything();
    };

    tStep steps_per_average = 10;

    std::string names[MAX_FUNC];


    void reset_names();
    void reset_elapsed();
    void reset_average();
    void reset_everything();


    void start_epoch();

    double time_now();
    std::string get_time_now_as_string();

    double check(int, int, double, std::string);



    void set_average_steps(tStep);


    std::string averagePerFunction(tStep step, int block, int stream, std::string name, double start_time, double elapsed);

    std::string averageAllFunctions(tStep step);
    void printAverageAllFunctions(tStep step);

    std::string timeLeft(tStep step, tStep num_steps, double print_time_left);




};






#endif /* timer_hpp */
