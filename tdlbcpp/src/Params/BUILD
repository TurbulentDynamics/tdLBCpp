cc_library(
    name = "pjson",
    srcs = ["jsoncpp.cpp"],
    hdrs = ["json.h"]
)


cc_library(
    name = "params",
    hdrs = ["Flow.hpp", "Grid.hpp", "Running.hpp", "BinFile.hpp", "ComputeUnitParams.hpp"],
    deps = [":pjson"],
    visibility = ["//tdlbcpp/src:__pkg__" ,"//tdlbcpp/tests/Params:__pkg__"],
)









