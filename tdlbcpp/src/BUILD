cc_library(
    name = "cxxopts",
    hdrs = ["cxxopts.hpp"]
)


cc_library(
    name = "computeUnit",
    hdrs = ["ComputeUnit.h", "Header.h", "Output.hpp", "DiskOutputTree.h", "Boundary.hpp", "CollisionEntropic.hpp", "GlobalStructures.hpp", "StreamingEsoteric.hpp", "cxxopts.hpp", "CollisionEgglesSomers.hpp", "ComputeUnit.hpp", "Forcing.hpp", "QVec.hpp", "StreamingNieve.hpp", "timer.h"],
    visibility = ["//tdlbcpp/tests/Params:__pkg__", "//tdlbcpp/tests:__pkg__"],
)

cc_binary(
    name = "tdlbcpp",
    srcs = ["main.cpp", "timer.cpp"],
    deps = [":cxxopts", "//tdlbcpp/src/Params:params", ":computeUnit",
            "//tdLBGeometryRushtonTurbineLib/Sources/tdLBGeometryRushtonTurbineLibCPP:tdLBGeometryRushtonTurbineLib"],
)








