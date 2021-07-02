cc_library(
    name = "utils",
    hdrs = ["utils.hpp"],
    srcs = ["utils.cpp"],
    visibility = ["//tdlbcpp/tests/Params:__pkg__"],
    deps = ["@googletest//:gtest_main"],
)

cc_test(
  name = "tests",
  srcs = ["DiskOutputTreeTest.cpp"],
  deps = [
          "//tdlbcpp/src/Params:params",
          "//tdlbcpp/src:computeUnit",
          "//tdlbcpp/tests:utils",
          "//tdlbcpp/tests/Params:ParamsCommon",
          "@googletest//:gtest_main"
  ],
  copts = ["-Itdlbcpp/src"]
)