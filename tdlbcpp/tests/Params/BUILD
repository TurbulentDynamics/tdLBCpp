cc_test(
  name = "tests",
  srcs = ["BinFileTest.cpp", "CheckpointTest.cpp", "ComputeUnitParamsTest.cpp", "OutputParamsTest.cpp", "FlowTest.cpp"],
  deps = [
          "//tdlbcpp/src/Params:params",
          "//tdlbcpp/src:computeUnit",
          "//tdlbcpp/tests:utils",
          "@googletest//:gtest_main"
  ],
  copts = ["-Itdlbcpp/src"]
)



