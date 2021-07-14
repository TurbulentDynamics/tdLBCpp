cc_library(
    name = "utils",
    hdrs = ["utils.hpp"],
    srcs = ["utils.cpp"],
    visibility = ["//tdlbcpp/tests/Params:__pkg__"],
    deps = ["@googletest//:gtest_main"],
)

cc_test(
  name = "tests",
  srcs = ["DiskOutputTreeTest.cpp", "StreamingNieveTest.cpp", "StreamingNieveTest.hpp"],
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