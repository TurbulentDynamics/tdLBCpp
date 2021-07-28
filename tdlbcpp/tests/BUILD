cc_library(
    name = "utils",
    hdrs = ["utils.hpp"],
    srcs = ["utils.cpp"],
    visibility = ["//visibility:public"],
    deps = ["@googletest//:gtest_main"],
)

cc_test(
  name = "tests",
  srcs = ["DiskOutputTreeTest.cpp", "StreamingNieveTest.cpp", "StreamingNieveTest.hpp", "ComputeUnitCheckpointTest.cpp", "ComputeUnitMemoryTest.cpp"],
  deps = [
          "//tdlbcpp/src/Params:params",
          "//tdlbcpp/src/Tools:toojpeg",
          "//tdlbcpp/src:computeUnit",
          "//tdlbcpp/tests:utils",
          "//tdlbcpp/tests/Params:ParamsCommon",
          "@tdLBGeometryRushtonTurbineLib//Sources/tdLBGeometryRushtonTurbineLibCPP:tdLBGeometryRushtonTurbineLib",
          "@googletest//:gtest_main"
  ],
  copts = ["-Itdlbcpp/src", "-Itdlbcpp/src/Tools"]
)
