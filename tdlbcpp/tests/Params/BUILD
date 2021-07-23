cc_library(
    name = "ParamsCommon",
    hdrs = ["ParamsCommon.hpp"],
    srcs = ["ParamsCommon.cpp"],
    visibility = ["//visibility:public"],
    deps = ["//tdlbcpp/src/Params:params",
            "//tdlbcpp/src/Tools:toojpeg",
            "@tdLBGeometryRushtonTurbineLib//Sources/tdLBGeometryRushtonTurbineLibCPP:tdLBGeometryRushtonTurbineLib",
            "//tdlbcpp/src:computeUnit",
            "//tdlbcpp/tests:utils", 
            "@googletest//:gtest_main"],
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



