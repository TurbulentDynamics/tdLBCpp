load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Google Test - Updated to latest stable version
http_archive(
    name = "googletest",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz"],
    strip_prefix = "googletest-1.15.2",
    sha256 = "7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926",
)

local_repository(
    name = "rules_cuda",
    path = __workspace_dir__ + "/third_party/rules_cuda",
)

load("@rules_cuda//cuda:dependencies.bzl", "rules_cuda_dependencies")
rules_cuda_dependencies()

load("@rules_cc//cc:repositories.bzl", "rules_cc_toolchains")
rules_cc_toolchains()

local_repository(
    name = "tdLBGeometryRushtonTurbineLib",
    path = __workspace_dir__ + "/tdLBGeometryRushtonTurbineLib",
)

local_repository(
    name = "gyb",
    path = __workspace_dir__ + "/third_party/gyb",
)
