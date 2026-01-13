load("@rules_cuda//cuda:defs.bzl", "cuda_library")

config_setting(
    name = "use_gpu",
    define_values = {"gpu" : "yes"},
)

config_setting(
    name = "use_mpi_gpu",
    define_values = {"mpi_gpu" : "yes"},
)

cc_library(
    name = "utils",
    hdrs = ["utils.hpp"],
    srcs = ["utils.cpp"],
    visibility = ["//visibility:public"],
    deps = ["@googletest//:gtest_main"],
)

testDeps = [
          "//tdlbcpp/src/Params:params",
          "//tdlbcpp/src/Tools:toojpeg",
          "//tdlbcpp/src:computeUnit",
          "//tdlbcpp/tests:utils",
          "@tdLBGeometryRushtonTurbineLib//Sources/tdLBGeometryRushtonTurbineLibCPP:tdLBGeometryRushtonTurbineLib",
          "@googletest//:gtest_main"
  ]
testCOpts = ["-Itdlbcpp/src", "-Itdlbcpp/src/Tools"]

cc_test(
  name = "diskOutputTest",
  srcs = ["DiskOutputTreeTest.cpp", "StreamingSimpleTest.cpp", "StreamingSimpleTest.hpp", "StreamingSimplePushTest.hpp", "ComputeUnitCheckpointTest.cpp", "ComputeUnitMemoryTest.cpp"],
  deps = testDeps + ["//tdlbcpp/tests/Params:ParamsCommon"],
  copts = testCOpts
)


testCOptsGpu = testCOpts + ["-Iexternal/local_cuda/cuda/samples/common/inc/"]

cuda_library(
  name = "gpuCudaCode",
  srcs = ["StreamingSimpleGpuTestLib.cpp"],
  deps = testDeps + ["//tdlbcpp/src:computeUnitCu", "//tdlbcpp/tests/Params:ParamsCommonCu"],
  copts = testCOptsGpu,
  target_compatible_with = select({
        ":use_gpu": [], ":use_mpi_gpu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
)

cc_binary(
  name = "gpuTest",
  srcs = ["StreamingSimpleGpuTest.cpp", "StreamingSimplePushTest.hpp"],
  deps = testDeps + [":gpuCudaCode", "@rules_cuda//cuda:cuda_runtime"],
  features = ["cuda", "-use_header_modules", "-layering_check", "-parse_headers"],
  copts = testCOptsGpu,
  target_compatible_with = select({
        ":use_gpu": [], ":use_mpi_gpu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
)
