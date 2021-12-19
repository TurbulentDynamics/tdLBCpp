cc_library(
    name = "pjson",
    srcs = ["jsoncpp.cpp"],
    hdrs = ["json.h"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "",
)


cc_library(
    name = "params",
    hdrs = ["CheckpointParams.hpp", "OutputParams.hpp", "FlowParams.hpp", "GridParams.hpp", "RunningParams.hpp", "BinFileParams.hpp", "ComputeUnitParams.hpp", "OrthoPlane.hpp", "Volume.hpp", "Angle.hpp", "PlaneAtAngle.hpp", "Sector.hpp"],
    deps = [":pjson"],
    visibility = ["//visibility:public"],
    include_prefix = "Params",
)









