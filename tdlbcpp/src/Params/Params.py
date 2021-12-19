#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Generate sample input files
"""
import json
from string import Template

__author__ = "Niall Ó Broin"
__copyright__ = "Copyright 2019, TURBULENT DYNAMICS"


EXTENSION = ".hpp"


WRITE_AS_JSON_TYPE = {
    "tNi": "Json::UInt64",
    "tStep": "Json::UInt64",
    "std::string": "std::string",
    "T": "double",
    "bool": "bool",
    "double": "double",
    "int": "int"
}

LOAD_FROM_JSON_AS = {
    "tNi": "UInt64",
    "tStep": "UInt64",
    "std::string": "String",
    "T": "Double",
    "bool": "Bool",
    "double": "Double",
    "int": "Int"
}

PYTHON_TYPE = {
    "tNi": int,
    "tStep": int,
    "std::string": str,
    "T": float,
    "bool": bool,
    "double": float,
    "int": int
}


class Param:
    def __init__(self, typeAlias, param_name, default, *note):
        self.param_name = param_name
        self.typeAlias = typeAlias
        self.val = PYTHON_TYPE[typeAlias](default)
        if typeAlias == "bool" and default == "false":
            self.val = False

        self.note = " ".join(note)

        self.loadFromJsonAs = "as" + LOAD_FROM_JSON_AS[self.typeAlias]

        self.writeAsJsonType = WRITE_AS_JSON_TYPE[self.typeAlias]


class ParamsBase:
    def __init__(self, raw_data):


        self.struct_name = self.__class__.__name__
        self.doc_string = self.__class__.__doc__ or "//"
        self.extra_methods = ""
        self.template = ""
        self.include = ""

        for line in raw_data.splitlines():
            if line == "":
                continue

            data = line.strip().split(" ")

            if len(data) < 3:
                print("Line wrong length: ", data)
                exit()

            p = Param(*data)
            setattr(self, p.param_name + "_obj", p)
            setattr(self, p.param_name, p.val)


    def update_val_and_get_param_objs(self):

        params_objs = list()

        params = [i for i in vars(self) if i.endswith("_obj")]
        for param_obj_name in params:

            param_name = param_obj_name[:-4]
            param_obj = getattr(self, param_obj_name)

            # Get val from self.param_name
            val = getattr(self, param_name)

            # Update the value on the param_obj
            setattr(param_obj, param_name, val)

            params_objs.append(param_obj)

        return params_objs


    @property
    def define(self):

        txt = ""

        for param_obj in self.update_val_and_get_param_objs():

            if param_obj.note:
                txt += "\n    //{note}\n".format(**vars(param_obj))

            if param_obj.typeAlias == "std::string":
                txt += "    {typeAlias} {param_name} = \"{val}\";\n".format_map(vars(param_obj))



            elif param_obj.typeAlias == "bool":
                txt += "    {typeAlias} {param_name} = ".format_map(vars(param_obj))
                if param_obj.val == True:
                    txt +=  "true;\n"
                else:
                    txt += "false;\n"

            else:
                txt += "    {typeAlias} {param_name} = {val};\n".format_map(vars(param_obj))

        return txt


    @property
    def loadJson(self):
        #step = (tStep)jsonParams["step"].asUInt64();

        txt = ""

        for param_obj in self.update_val_and_get_param_objs():

            txt += "    {param_name} = ({typeAlias})jsonParams[\"{param_name}\"].{loadFromJsonAs}();\n".format_map(vars(param_obj))

        return txt


    @property
    def saveJson(self):
        #jsonParams["ngx"] = (int)ngx;
        #jsonParams["step"] = (Json::UInt64)step;

        txt = ""

        for param_obj in self.update_val_and_get_param_objs():

            txt += "    jsonParams[\"{param_name}\"] = ({writeAsJsonType}){param_name};\n".format_map(vars(param_obj))

        return txt


    @property
    def json(self):
        json = dict()

        for param_obj in self.update_val_and_get_param_objs():

            json[param_obj.param_name] = getattr(self, param_obj.param_name)

        return json



#    @property
    def cpp_file(self):
        with open("Params_Template.py") as f:
            s = Template(f.read())

        data = dict(**vars(self))
        data["define"] = self.define
        data["save_json"] = self.saveJson
        data["load_json"] = self.loadJson


        u = s.substitute(data)

        with open(self.struct_name + EXTENSION, 'w') as f:
            f.write(u)






class GridParams(ParamsBase):
    """
    //Grid Params are the maximum extent of the regular lattice.
    """

    def __init__(self):
        super().__init__("""
tNi ngx 1
tNi ngy 1
tNi ngz 1


tNi x 60
tNi y 60
tNi z 60

tNi multiStep 1
std::string strMinQVecPrecision float
""")



class FlowParams(ParamsBase):
    def __init__(self):
        super().__init__("""

T initialRho 8.0
T reMNonDimensional 7300.0

T uav 0.1

T cs0 0.12 ratio mixing length / lattice spacing delta (Smagorinsky)

T g3 0.8 compensation of third order terms

T nu 0.0 kinematic viscosity

T fx0 0.0 forcing in x-direction


T Re_m 0.0 Reynolds number based on mean or tip velocity

T Re_f 0.0 Reynolds number based on the friction velocity uf


T uf 0.0 friction velocity

T alpha 0.97
T beta 1.9

bool useLES false

std::string collision EgglesSomers
std::string streaming Simple
""")

        self.template = "template <typename T>"

        self.extra_methods = """
    void calcNuAndRe_m(int impellerBladeOuterRadius){

        Re_m = reMNonDimensional * M_PI / 2.0;

        nu  = uav * (T)impellerBladeOuterRadius / Re_m;
    }

    FlowParams<double> asDouble(){
        FlowParams<double> f;

        f.initialRho = (double)initialRho;
        f.reMNonDimensional = (double)reMNonDimensional;
        f.uav = (double)uav;
        f.cs0 = (double)cs0;
        f.g3 = (double)g3;
        f.nu = (double)nu;
        f.fx0 = (double)fx0;
        f.Re_m = (double)Re_m;
        f.Re_f = (double)Re_f;
        f.uf = (double)uf;

        f.alpha = (double)alpha;
        f.beta = (double)beta;

        f.useLES = useLES;
        f.collision = collision;
        f.streaming = streaming;

        return f;
    }
"""




class RunningParams(ParamsBase):
    def __init__(self):
        super().__init__("""
tStep step 1
double angle 0
tStep num_steps 20
tStep impellerStartupStepsUntilNormalSpeed 30

std::string runningDataFileDir .
std::string runningDataFilePrefix debug_running
tStep numStepsForAverageCalc 10
tStep repeatPrintTimerToFile 20
tStep repeatPrintTimerToStdOut 10

""")

        self.extra_methods = """
    void update(tStep _step, double _angle){

        step = (tStep)_step;
        angle = (double)_angle;

    }

    void incrementStep(){
        step ++;
    }
"""


class BinFileParams(ParamsBase):
    def __init__(self):
        super().__init__("""
std::string filePath NoFilePath
std::string name NoName
std::string note NoNote


std::string structName tDisk_colrow_Q4
tNi binFileSizeInStructs 0

std::string coordsType uint16_t
bool hasGridtCoords false
bool hasColRowtCoords true

std::string reference absolute
tNi i0 0
tNi j0 0
tNi k0 0

std::string QDataType float
int QOutputLength 4
""")

class CheckpointParams(ParamsBase):
    def __init__(self):
        super().__init__("""

bool startWithCheckpoint false
std::string checkpointLoadFromDir notSet

int checkpointRepeat 10
std::string checkpointWriteRootDir .
std::string checkpointWriteDirPrefix debug
bool checkpointWriteDirAppendTime true

""")


class ComputeUnitParams(ParamsBase):
    def __init__(self):
        super().__init__("""
int nodeID 0
int deviceID 0

int idi 0
int idj 0
int idk 0

tNi x 0
tNi y 0
tNi z 0

tNi i0 0
tNi j0 0
tNi k0 0

tNi ghost 0
tNi resolution 0
std::string strQVecPrecision notSet
std::string strMemoryLayout notSet
""")

        self.extra_methods = """
    ComputeUnitParams() {}

    ComputeUnitParams(int nodeID, int deviceID, int idi, int idj, int idk, tNi x, tNi y, tNi z, tNi i0, tNi j0, tNi k0, tNi ghost)
     : nodeID(nodeID), deviceID(deviceID), idi(idi), idj(idj), idk(idk), x(x), y(y), z(z), i0(i0), j0(j0), k0(k0), ghost(ghost) {}
"""


class OrthoPlane(ParamsBase):
    def __init__(self):
        super().__init__("""
std::string name_root plane
std::string QDataType float
int Q_output_len 4
bool use_half_float false

tNi cutAt 0
tStep repeat 0
tStep start_at_step 0
tStep end_at_repeat 0
""")


class Volume(ParamsBase):
    def __init__(self):
        super().__init__("""
std::string name_root volume
tStep repeat 0

int Q_output_len 4
tStep start_at_step 0
tStep end_at_repeat 0
bool use_half_float false

std::string QDataType float
""")

class Angle(ParamsBase):
    def __init__(self):
        super().__init__("""
std::string name_root angle
tStep repeat 0
double degrees 0

int Q_output_len 4
tStep start_at_step 0
tStep end_at_repeat 0
bool use_half_float false

std::string QDataType float
""")

class PlaneAtAngle(ParamsBase):
    def __init__(self):
        super().__init__("""
std::string name_root plane_at_angle
double degrees 0
double tolerance 0
tNi cutAt 0

int Q_output_len 4
tStep start_at_step 0
tStep end_at_repeat 0
bool use_half_float false

std::string QDataType float
""")


class Sector(ParamsBase):
    def __init__(self):
        super().__init__("""
std::string name_root sector
tStep repeat 0

double angle_infront_blade 0.0
double angle_behind_blade 0.0

int Q_output_len 4
tStep start_at_step 0
tStep end_at_repeat 0
bool use_half_float false

std::string QDataType float
""")



class OutputParams(ParamsBase):
    def __init__(self):
        super().__init__("""
std::string outputRootDir debug_output_dir
""")

        self.ortho_plane_objs = list()
        self.volume_objs = list()
        self.angle_objs = list()
        self.plane_at_angle_objs = list()
        self.sector_objs = list()

        self.include = """
#include "OrthoPlane.hpp"
#include "Volume.hpp"
#include "Angle.hpp"
#include "PlaneAtAngle.hpp"
#include "Sector.hpp"
"""

    @property
    def json(self):

        json = dict()

        for param_obj in self.update_val_and_get_param_objs():
            json[param_obj.param_name] = getattr(self, param_obj.param_name)


        def upsert_level2(objs):
            for obj in objs:
                if obj.struct_name in json:
                    json[obj.struct_name].append(obj.json)
                else:
                    json[obj.struct_name] = [obj.json]


        upsert_level2(self.ortho_plane_objs)
        upsert_level2(self.volume_objs)
        upsert_level2(self.angle_objs)
        upsert_level2(self.plane_at_angle_objs)
        upsert_level2(self.sector_objs)

        return json


    def add_debug_output(self, grid):

        orthoPlaneXY = OrthoPlane()
        orthoPlaneXY.struct_name = "XY_planes"
        orthoPlaneXY.cutAt = grid.z / 2
        orthoPlaneXY.repeat = 10
        orthoPlaneXY.name_root = "plot_slice"

        self.ortho_plane_objs.append(orthoPlaneXY)

        orthoPlaneXZ = OrthoPlane()
        orthoPlaneXZ.struct_name = "XZ_planes"
        orthoPlaneXZ.cutAt = grid.y / 3 * 2
        orthoPlaneXZ.repeat = 10
        orthoPlaneXZ.name_root = "plot_axis"

        self.ortho_plane_objs.append(orthoPlaneXZ)





if __name__ == '__main__':



    grid = GridParams()
    grid.cpp_file()


    flow = FlowParams()
    flow.cpp_file()

    running = RunningParams()
    running.cpp_file()

    binfile = BinFileParams()
    binfile.cpp_file()

    chkp = CheckpointParams()
    chkp.cpp_file()

    cup = ComputeUnitParams()
    cup.cpp_file()



    out = OrthoPlane()
    out.cpp_file()

    out = Volume()
    out.cpp_file()

    out = Angle()
    out.cpp_file()

    out = PlaneAtAngle()
    out.cpp_file()

    out = Sector()
    out.cpp_file()



