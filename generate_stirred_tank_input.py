#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Generate sample input files
"""
import sys
import json
import argparse


__author__ = "Niall Ó Broin"
__copyright__ = "Copyright 2019, TURBULENT DYNAMICS"

sys.path.append('tdlbcpp/src/Params')
from Params import *



FILENAME = "input_debug_gridx60_numSteps20.json"

# (ngx, grid.x, numSteps, chkRepeat)
val = [
 (1, 9, 3, 0),
# (1, 60, 100, 20), Initialised Params Setup
 (1, 100, 200, 50),
 (1, 200, 1000, 500),
 (1, 400, 2000, 500),
 (1, 600, 5000, 1000),
 ]



def writeJsonFile(data, filename):

    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)


def printFile(filename):
    with open(filename, "r") as f:
        print(f.read())





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-f", "--filename", type=str, help="filanem")

    parser.add_argument("--esotwist", dest="esotwist", action="store_true", help="Streaming with Esotwist")

    parser.add_argument("-x", "--gridx", type=int, help="Set the grid.x size (y,z = x)")
    parser.add_argument("-n", "--num_steps", type=int, help="Run number of steps")
    parser.add_argument("-q", "--impellerStartupStepsUntilNormalSpeed", type=int, help="Start the impeller slowly")


    parser.add_argument("-p", "--checkpoint_repeat", type=int, help="Checkpoint repeat every n steps")
    parser.add_argument("-d", "--checkpointRootDir", type=str, help="Checkpoint Root Dir")
    parser.add_argument("-a", "--start_with_checkpoint", type=bool, help="Start with Checkpoint")
    parser.add_argument("-b", "--load_checkpoint_dirname", type=str, help="Load Checkpoint Dirname")

    parser.add_argument("--initialRho", type=float, help="Initial density")
    parser.add_argument("-r", "--reMNonDimensional", type=int, help="Reynolds number")
    parser.add_argument("--uav", type=float, help="X")

    parser.add_argument("-o", "--outputRootDir", type=int, help="Output Root Dir")


    parser.add_argument("-l", "--les", type=bool, help="Use Large Eddy Simulations")
    parser.add_argument("-s", "--cs0", type=float, help="Smorginsky")
    args = parser.parse_args()
    print(args)


    grid = GridParams()
    flow = FlowParams()
    running = RunningParams()
    binfile = BinFileParams()
    chkp = CheckpointParams()
    cup = ComputeUnitParams()
    out = OutputParams()


    if args.gridx:
        grid.x = args.gridx
        grid.y = args.gridx
        grid.z = args.gridx

    out.add_debug_output(grid)


    if args.esotwist:
        flow.streaming = "Esotwist"


    if args.num_steps:
        running.num_steps = args.num_steps
    if args.impellerStartupStepsUntilNormalSpeed:
        running.impellerStartupStepsUntilNormalSpeed = args.impellerStartupStepsUntilNormalSpeed

    if args.outputRootDir:
        out.outputRootDir = args.outputRootDir



    if args.initialRho:
        flow.initialRho = args.initialRho
    if args.reMNonDimensional:
        flow.reMNonDimensional = args.reMNonDimensional
    if args.uav:
        flow.uav = args.uav
    if args.les:
        flow.useLES = args.les
    if args.cs0:
        flow.cs0 = args.cs0




    json = dict()
    json[grid.struct_name] = grid.json
    json[flow.struct_name] = flow.json
    json[running.struct_name] = running.json
    json[binfile.struct_name] = binfile.json
    json[chkp.struct_name] = chkp.json
    json[cup.struct_name] = cup.json
    json[out.struct_name] = out.json


    writeJsonFile(json, args.filename or FILENAME)

    printFile(args.filename or FILENAME)





if __name__ == "__main__":
    main()






