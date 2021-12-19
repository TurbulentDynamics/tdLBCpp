//
//  timer.cpp
//
//  Created by Niall Ó Broin on 23/07/2018.
//  Copyright © 2018 Niall Ó Broin. All rights reserved.
//

//  Posix/Linux
#include <time.h>
#include <sys/time.h>

#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <chrono>
#include <fstream>


#if _OPENMP
#include <omp.h>
#endif

#include "timer.h"



double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}





void Multi_Timer::reset_names(){
    for (int i = 0; i < MAX_FUNC; i++){
        names[i] = "-";
        block[i] = -1;
        stream[i] = -1;
    }
}

void Multi_Timer::reset_elapsed(){
    for (int i = 0; i < MAX_FUNC; i++){
        elapsed[i] = 0.0;
    }
}

void Multi_Timer::reset_average(){
    for (int i = 0; i < MAX_FUNC; i++){
        average[i] = 0.0;
    }
}

void Multi_Timer::reset_everything(){
    reset_names();
    reset_elapsed();
    reset_average();
}


double Multi_Timer::get_now(){

#ifdef _OPENMP
    return omp_get_wtime();
#else
    return get_wall_time();
#endif
}






//https://stackoverflow.com/questions/16357999/current-date-and-time-as-string
std::string Multi_Timer::get_time_now_as_string(){

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



double Multi_Timer::time_now(){
    return get_now() - epoch;
}


void Multi_Timer::start_epoch(){
    epoch = get_now();
}



void Multi_Timer::set_average_steps(tStep step){

    steps_per_average = step;
}





std::string Multi_Timer::averagePerFunction(tStep step, int block, int stream, std::string name, double start_time, double elapsed){


    using namespace std;

    stringstream sstream;

    sstream << setfill('0') << setw(2);
    sstream << "Node " << rank << ": Block " << block << " Stream " << stream;

    sstream << " Step " << step << "  ";


    name = name.substr(0, 40);
    sstream << left << setfill('.') << setw(44) << name << setfill(' ');


    sstream << " start " << fixed << setw(7) << setprecision(4) << start_time;

    sstream << " elapsed " << fixed << setw(7) << setprecision(4) << elapsed << endl;


    return sstream.str();
}


std::string Multi_Timer::averageAllFunctions(tStep step){


    using namespace std;

    stringstream sstream;

    sstream << "Average time for the last " << steps_per_average << " steps @ Current step: " << step << "\n";

    for (int b = 0; b < MAX_FUNC; b++){
        if (block[b] == -1) break;

        for (int func = 0; func < MAX_FUNC; func++){

            if (block[func] == b && names[func] != "-"){


                sstream << averagePerFunction(step, block[func], stream[func], names[func], start_time[func], elapsed[func]);

            }
        }
    }

    return sstream.str();
}


void Multi_Timer::printAverageAllFunctions(tStep step) {


    if (step == 1 || (step > 1 && (step % steps_per_average) == 0)) {
        std::cout << averageAllFunctions(step);
    }

}

double Multi_Timer::check(int block_num, int stream_num, double start, std::string func){

    int next = -1;

    for (int f = 0; f < MAX_FUNC; f++){

        if (func == names[f]){
            average[f] += time_now() - start;
            elapsed[f] = time_now() - start;
            start_time[f] = start;
            break;
        }

        if (names[f] == "-"){
            next = f;
            break;
        }

    }



    if (next >= 0){
        names[next] = func;
        block[next] = block_num;
        stream[next] = stream_num;
        start_time[next] = start;

        average[next] = time_now() - start;
        elapsed[next] = time_now() - start;
    }

    return time_now();
}





std::string Multi_Timer::timeLeft(tStep step, tStep num_steps, double print_time_left) {

    using namespace std;

    double time_left = (time_now() - print_time_left) * (num_steps - step) / 60;

    stringstream sstream;

    sstream << "This step " << fixed << setw(4) << setprecision(1) << (time_now() - print_time_left) << "s ";

    sstream << "    Remaining mins: " << fixed << setw(4) << setprecision(1) << time_left << endl;

    return sstream.str();
}










