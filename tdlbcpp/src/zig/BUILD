load("//tdlbcpp/src:targets.bzl", "zig_lib")

zig_lib(
    name = "zigModules",
    srcs = ["ComputeUnitIntf.zig"],
    deps = ["//tdlbcpp/src:header"],
    hdrs = ["ComputeUnit.zig", "Boundary.zig", "CollisionEgglesSomers.zig", "QVec.zig", "GeomPolar.zig"],
    visibility = ["//visibility:public"],
    copts = ["-I", "tdlbcpp/src"],
)