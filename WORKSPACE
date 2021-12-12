load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
  name = "googletest",
  remote = "https://github.com/google/googletest",
  tag = "release-1.11.0"
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
