cc_library(
    name = "pjson",
    srcs = ["jsoncpp.cpp"],
    hdrs = ["json.h"],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "params",
    hdrs = ["Checkpoint.hpp", "OutputParams.hpp", "Flow.hpp", "Grid.hpp", "Running.hpp", "BinFile.hpp", "ComputeUnitParams.hpp"],
    deps = [":pjson"],
    visibility = ["//visibility:public"],
)









