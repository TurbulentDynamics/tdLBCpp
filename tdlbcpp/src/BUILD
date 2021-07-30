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

cc_binary(
    name = "tdlbcpp",
    srcs = ["main.cpp"],
    deps = [":cxxopts", ":timer", "//tdlbcpp/src/Params:params", ":computeUnit", "//tdlbcpp/src/Tools:toojpeg",
            "@tdLBGeometryRushtonTurbineLib//Sources/tdLBGeometryRushtonTurbineLibCPP:tdLBGeometryRushtonTurbineLib"],
)








