load("@rules_cuda//cuda:defs.bzl", "cuda_library")

config_setting(
    name = "use_gpu",
    define_values = {"gpu" : "yes"},
)

config_setting(
    name = "use_mpi_gpu",
    define_values = {"mpi_gpu" : "yes"},
)

paramsCommonDeps = ["//tdlbcpp/src/Params:params",
            "//tdlbcpp/src/Tools:toojpeg",
            "@tdLBGeometryRushtonTurbineLib//Sources/tdLBGeometryRushtonTurbineLibCPP:tdLBGeometryRushtonTurbineLib",
            "//tdlbcpp/src:computeUnit",
            "//tdlbcpp/tests:utils", 
            "@googletest//:gtest_main"]

cc_library(
    name = "ParamsCommon",
    hdrs = ["ParamsCommon.hpp"],
    srcs = ["ParamsCommon.cpp"],
    visibility = ["//visibility:public"],
    deps = paramsCommonDeps,
    copts = ["-Itdlbcpp/src"],
    target_compatible_with = select({
        ":use_gpu": ["@platforms//:incompatible"], ":use_mpi_gpu": ["@platforms//:incompatible"],
        "//conditions:default": [],
    })
)

cuda_library(
    name = "ParamsCommonCu",
    hdrs = ["ParamsCommon.hpp"],
    srcs = ["ParamsCommon.cpp"],
    visibility = ["//visibility:public"],
    deps = paramsCommonDeps + ["//tdlbcpp/src:computeUnitCu"],
    copts = ["-Iexternal/local_cuda/cuda/samples/common/inc/", "-Itdlbcpp/src"],
    target_compatible_with = select({
        ":use_gpu": [], ":use_mpi_gpu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
)

cc_test(
  name = "tests",
  srcs = ["BinFileTest.cpp", "CheckpointTest.cpp", "ComputeUnitParamsTest.cpp", "OutputParamsTest.cpp", "FlowTest.cpp", "GridTest.cpp", "RunningTest.cpp"],
  deps = [
          "//tdlbcpp/src/Params:params",
          "//tdlbcpp/src:computeUnit",
          "//tdlbcpp/tests:utils",
          "//tdlbcpp/tests/Params:ParamsCommon",
          "@googletest//:gtest_main"
  ],
  copts = ["-Itdlbcpp/src"]
)



