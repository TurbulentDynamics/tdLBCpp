load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("//tdlbcpp/src:targets.bzl", "cc_binary_rule")

config_setting(
    name = "use_gpu",
    define_values = {"gpu" : "yes"},
)

config_setting(
    name = "use_mpi",
    define_values = {"mpi" : "yes"},
)

config_setting(
    name = "use_mpi_gpu",
    define_values = {"mpi_gpu" : "yes"},
)

cc_library(
    name = "cxxopts",
    hdrs = ["cxxopts.hpp"]
)

cc_library(
    name = "timer",
    srcs = ["timer.cpp"],
    hdrs = ["timer.h"],
    deps = [":computeUnit"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "computeUnit",
    hdrs = ["ComputeUnit.h", "Header.h", "Output.hpp", "DiskOutputTree.h", "Boundary.hpp", "CollisionEntropic.hpp", "GlobalStructures.hpp", "StreamingEsoteric.hpp", "cxxopts.hpp", "CollisionEgglesSomers.hpp", "ComputeUnit.hpp", "ComputeUnitOutput.hpp", "Forcing.hpp", "QVec.hpp", "StreamingNieve.hpp", "timer.h", "Field.hpp"],
    visibility = ["//visibility:public"],
    strip_include_prefix="",
)

cc_library(
    name = "computeUnitCu",
    hdrs = ["ComputeUnit.cuh", "ComputeUnitGpu.h", "ComputeUnitGpu.hpp", "Forcing.cuh"],
    visibility = ["//visibility:public"],
)

tdlbcpp_deps = [":cxxopts", ":timer", "//tdlbcpp/src/Params:params", ":computeUnit", "//tdlbcpp/src/Tools:toojpeg",
                "@tdLBGeometryRushtonTurbineLib//Sources/tdLBGeometryRushtonTurbineLibCPP:tdLBGeometryRushtonTurbineLib"]

cc_binary_rule(
    name = "tdlbcpp",
    srcs = ["main.cpp"],
    deps = tdlbcpp_deps,
    deps_gpu = tdlbcpp_deps + [":computeUnitCu"],
    copts = ["-Iexternal/local_cuda/cuda/samples/common/inc/"],
)








