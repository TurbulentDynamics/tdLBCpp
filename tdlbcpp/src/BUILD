cc_library(
    name = "json",
    srcs = ["jsoncpp.cpp"],
    hdrs = ["json.h"]
)

cc_library(
    name = "cxxopts",
    hdrs = ["cxxopts.hpp"]
)


cc_library(
    name = "computeUnit",
    hdrs = ["BaseParams.h", "ComputeUnit.h", "Header.h", "OutputConfig.h", "PlotDir.h", "PlotDirMeta.h", "QVecBinMeta.h", "Boundary.hpp", "CollisionEntropic.hpp", "FlowParams.hpp", "GlobalStructures.hpp", "Output.hpp", "StreamingEsoteric.hpp", "cxxopts.hpp", "CollisionEgglesSomers.hpp", "ComputeUnit.hpp", "Forcing.hpp", "GridParams.hpp", "QVec.hpp", "StreamingNieve.hpp"],
)

cc_binary(
    name = "tdlbcpp",
    srcs = ["main.cpp"],
    deps = [":json", ":cxxopts", ":computeUnit",
            "//tdLBGeometryRushtonTurbineLib/Sources/tdLBGeometryRushtonTurbineLibCPP:tdLBGeometryRushtonTurbineLib"],
)








