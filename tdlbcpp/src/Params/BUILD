cc_library(
    name = "pjson",
    srcs = ["jsoncpp.cpp"],
    hdrs = ["json.h"],
    visibility = ["//visibility:public"],
    strip_include_prefix = "",
)


cc_library(
    name = "params",
    hdrs = ["Checkpoint.hpp", "OutputParams.hpp", "Flow.hpp", "Grid.hpp", "Running.hpp", "BinFile.hpp", "ComputeUnitParams.hpp"],
    deps = [":pjson"],
    visibility = ["//visibility:public"],
    include_prefix = "Params",
)









