
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
#include <iomanip>
#include <vector>
#include <cstdint>
#include <map>
#include <string>
#include <time.h>
#include "json.h"


#include "Header.h"

#include "Params/GridParams.hpp"
#include "Params/ComputeUnitParams.hpp"
#include "Params/FlowParams.hpp"





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

    //Need to sync this across nodes
    std::string initTime = "initTimeNotSet";

    Json::Value cuJson;

    Json::Value grid;
    Json::Value flow;
    Json::Value running;
    Json::Value output;
    Json::Value checkpoint;




public:

    //These directories maybe on separate external disks
    std::string outputDir = "Error_UnInitialised_Output_Dir";

    std::string checkpointWriteDir = "Error_UnInitialised_Checkpoint_Dir";

    std::string runningDataPath = "./Error_UnInitialised_Running_Data_Dir";



    DiskOutputTree(OutputParams o, CheckpointParams c){

        initTime = getTimeNowAsString();

        output = o.getJson();
        setOutputDir();
        createDir(outputDir);

        checkpoint = c.getJson();
        setCheckpointWriteDir();
        createDir(checkpointWriteDir);
    };


    DiskOutputTree(ComputeUnitParams cu, Json::Value grid, Json::Value flow, Json::Value running, Json::Value output, Json::Value checkpoint): cu(cu),  grid(grid), flow(flow), running(running), output(output), checkpoint(checkpoint) {

        initTime = getTimeNowAsString();

        cuJson = cu.getJson();

        setOutputDir();
        createDir(outputDir);

        setCheckpointWriteDir();
        createDir(checkpointWriteDir);


    };


    DiskOutputTree(ComputeUnitParams cu1, GridParams grid1, FlowParams<double> flow1, RunningParams running1,  OutputParams output1, CheckpointParams checkpoint1){

        initTime = getTimeNowAsString();

        cu = cu1;

        cuJson = cu1.getJson();

        grid = grid1.getJson();
        flow =  flow1.getJson();
        running = running1.getJson();
        output = output1.getJson();
        checkpoint = checkpoint1.getJson();


        setOutputDir();
        createDir(outputDir);

        setCheckpointWriteDir();
        createDir(checkpointWriteDir);
    };

    template <typename T>
    void setParams(ComputeUnitParams cu1, GridParams grid1, FlowParams<T> flow1, RunningParams running1, OutputParams output1, CheckpointParams checkpoint1){

        cu = cu1;
        cuJson = cu1.getJson();

        grid = grid1.getJson();
        flow =  flow1.getJson();
        running = running1.getJson();
        output = output1.getJson();
        checkpoint = checkpoint1.getJson();

    }

    void setComputeUnitParams(ComputeUnitParams cu1) {
        cu = cu1;
        cuJson = cu1.getJson();
    }

    void setGridParams(GridParams grid1){
        grid = grid1.getJson();
    }

    template <typename T>
    void setFlowParams(FlowParams<T> flow1) {
        flow =  flow1.getJson();
    }

    void setRunningParams(RunningParams running1){
        running = running1.getJson();
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

    inline bool dirExists(std::string path) {
        return pathExists(path);
    }


    void createDir(std::string dir){

        std::cout << "DiskOutputTree creating directory: " << dir << std::endl;
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





    std::string paramSummary(){
        std::string str = "_gridx_" + grid["x"].asString();
        str += "_re_" + flow["reMNonDimensional"].asString();
//        str += "_les_" + flow["useLES"].asString();
//        str += "_uav_" + flow["uav"].asString();
        return str;
    }

    std::string getInitTimeAndParams(std::string prefix = "", tStep step = 0){

        std::string str = "";
        if (prefix != "") str += prefix;
        if (step) str += "_step_" + formatStep(step);

        str += "_" + initTime;
        str += paramSummary();

        return str;
    }

    std::string getTimeNowAndParams(std::string prefix = "", tStep step = 0){

        std::string str = "";
        if (prefix != "") str += prefix + "_";
        if (step) str += "step_" + formatStep(step) + "_";

        str += getTimeNowAsString() + "_";
        str += paramSummary();

        return str;
    }






    void setRunningDataFile(std::string dir, std::string fileName){

        if (!dirExists(dir)){
            createDir(dir);
        }

        runningDataPath = dir + "/" + fileName;

    }


    std::string nodePrintText(std::string text){

        std::stringstream nodeText;

        nodeText << "Node " << cuJson["nodeID"] << ": ";

        nodeText << text;

        return nodeText.str();
    }


    void writeToRunningDataFile(std::string text){

        using namespace std;

        ofstream runningDataFile;


        if (pathExists(runningDataPath)){
            runningDataFile.open(runningDataPath, ofstream::app);
        } else {
            runningDataFile.open(runningDataPath, ofstream::out);
        }

        if (runningDataFile.is_open()){
            runningDataFile << text;
        } else {

            cout << "Error: runningFile not open ";
            cout << "Trying to print -> " << text << "<-" << endl;
        }
    }


    void writeToRunningDataFileAndPrint(std::string text){


        writeToRunningDataFile(nodePrintText(text));
        std::cout << nodePrintText(text);
    }


    //==================================================



    inline std::string formatStep(tStep step){

        std::stringstream sstream;

        sstream << std::setw(8) << std::setfill('0') << patch::to_string(step);

        return sstream.str();
    }

    void setOutputDir(){

        outputDir = output["outputRootDir"].asString() + "_" + initTime;
    }



    std::string formatPlotDir(std::string prefix, std::string plotType, tStep step) {
        return outputDir + "/" + prefix + "." + plotType + ".V5.step_" + formatStep(step);
    }

    std::string formatXYPlaneDir(tStep step, tNi atK, const std::string prefix="plot"){

        return formatPlotDir(prefix, "xyPlane", step) + ".cut_" + patch::to_string(atK);
    }

    //Formally Axis
    std::string formatXZPlaneDir(tStep step, tNi atJ, const std::string prefix="plot"){

        // FIXME:

        return formatPlotDir(prefix, "xzPlane", step) + ".cut_" + patch::to_string(atJ);
    }

    //Formally Slice
    std::string formatYZPlaneDir(tStep step, tNi atI, const std::string prefix="plot"){

        return formatPlotDir(prefix, "yzPlane", step) + ".cut_" + patch::to_string(atI);
    }

    std::string formatVolumeDir(tStep step, const std::string prefix="plot"){
        return formatPlotDir(prefix, "volume", step);
    }


    std::string formatCaptureAtBladeAngleDir(tStep step, int angle, int bladeId, const std::string prefix="plot"){

        return formatPlotDir(prefix, "rotational_capture", step) + ".angle_" + patch::to_string(angle) + ".bladeId_" + patch::to_string(bladeId);
    }


    std::string formatAxisWhenBladeAngleDir(tStep step, int angle, const std::string prefix="plot"){

        return formatPlotDir(prefix, "yzAnglePlane", step) + ".angle_" + patch::to_string(angle);
    }



    std::string formatRotatingSectorDir(tStep step, int angle, const std::string prefix="plot"){

        std::string ret = "";
        return ret;
    }


    inline std::string formatId(){
        return patch::to_string(cu.idi) + "." + patch::to_string(cu.idj) + "." + patch::to_string(cu.idk);
    }

    inline std::string formatCUid(){
        return "CUid." + formatId();
    }


    std::string formatQVecBinFileNamePath(std::string dir){
        return dir + "/" + formatCUid() + ".V5.QVec.bin";
    }

    std::string formatF3BinFileNamePath(std::string dir){
        return dir + "/" + formatCUid() + ".V5.F3.bin";
    }

    std::string formatJpegFileNamePath(std::string dir){
        return dir + "/" + formatCUid() + ".V5.jpeg";
    }



    Json::Value getJsonParams(BinFileParams binFormat, RunningParams runParam){
        Json::Value jsonParams;

        try {

            jsonParams["ComputeUnitParams"] = cuJson;

            jsonParams["BinFileParams"] = binFormat.getJson();
            jsonParams["GridParams"] = grid;
            jsonParams["FlowParams"] = flow;
            jsonParams["RunningParams"] = runParam.getJson();
            jsonParams["OutputParams"] = output;
            jsonParams["CheckpointParams"] = checkpoint;

        } catch(std::exception& e) {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return 1;
        }
        return jsonParams;
    }


    int writeAllParamsJson(BinFileParams binFormat, RunningParams runParam, std::string filePath){

        Json::Value jsonParams = getJsonParams(binFormat, runParam);

        try {
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams;
            out.close();

            return 0;
        }
        catch(std::exception& e) {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return 1;
        }

        return 0;
    }

    int writeAllParamsJson(BinFileParams binFormat, RunningParams runParam){

        Json::Value jsonParams = getJsonParams(binFormat, runParam);

        try {

            std::string filePath = binFormat.filePath + ".json";

            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << jsonParams;
            out.close();

            return 0;
        }
        catch(std::exception& e) {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
            return 1;
        }

        return 0;
    }




    int readAllParamsJson(std::string filePath, BinFileParams format, RunningParams runParam){


        try {
            std::ifstream in(filePath.c_str());
            Json::Value jsonParams;
            in >> jsonParams;
            in.close();


            // TODO:


            cuJson = jsonParams["ComputeUnitParams"];
            cu.getParamsFromJson(cuJson);

            //Not necessary atm
            //            binFileJson = jsonParams["BinFileParams"];

            grid = jsonParams["GridParams"];
            flow = jsonParams["FlowParams"];
            running = jsonParams["RunningParams"];
            output = jsonParams["OutputParams"];
            checkpoint = jsonParams["CheckpointParams"];



        }
        catch(std::exception& e)
        {
            std::cerr << "Unhandled Exception reached parsing arguments: "
            << e.what() << ", application will now exit" << std::endl;
        }



        return 0;

    }

    template <typename T>
    FlowParams<T> getFlowParams() {
        FlowParams<T> flowParams;
        if (!flow.empty()) {
            flowParams.getParamsFromJson(flow);
        }
        return flowParams;
    }

    ComputeUnitParams getComputeUnitParams() {
        return cu;
    }

    //===================================




    std::string checkpointName(){

        std::string name = "";
        if (checkpoint["checkpointWriteDirPrefix"].asString() != "") name += checkpoint["checkpointWriteDirPrefix"].asString() + "_";
        name += "checkpoint";
        if (checkpoint["checkpointWriteDirAppendTime"].asBool()) name += "_" + initTime;

        return name;
    }

    void setCheckpointWriteDir(){

        checkpointWriteDir = checkpoint["checkpointWriteRootDir"].asString();

        checkpointWriteDir += "/" + checkpointName();

    }

    std::string getCheckpointDirName(RunningParams run, bool create=true){


        std::string dirName = checkpointWriteDir + "/";

        dirName += checkpointName() + "_step_" + std::to_string(run.step);

        if (create) createDir(dirName);

        return dirName;
    }

    std::string getCheckpointFilePath(std::string dir, std::string unit_name, std::string matrix){


        std::string path = dir + "/checkpoint_grid." + formatId() + ".";

        path += unit_name + "." + matrix;

        return path;
    }

    std::string getAllParamsFilePath(std::string dir, std::string unit_name){


        std::string path = dir + "/AllParams." + formatId() + ".";

        path += unit_name;

        return path;
    }





};


#endif


