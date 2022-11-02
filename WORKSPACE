load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


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

local_repository(
    name = "zig_integration",
    path = __workspace_dir__ + "/third_party/zig_integration",
)

BAZEL_ZIG_CC_VERSION = "v1.0.0-rc3"

http_archive(
    name = "bazel-zig-cc",
    #sha256 = "73afa7e1af49e3dbfa1bae9362438cdc51cb177c359a6041a7a403011179d0b5",
    strip_prefix = "bazel-zig-cc-{}".format(BAZEL_ZIG_CC_VERSION),
    urls = ["https://git.sr.ht/~motiejus/bazel-zig-cc/archive/{}.tar.gz".format(BAZEL_ZIG_CC_VERSION)],
    patch_args = ["-p1"],
    patches = [
        "@zig_integration//:bazel_zig_cc_{}.patch".format(BAZEL_ZIG_CC_VERSION),
    ],
)

load("@bazel-zig-cc//toolchain:defs.bzl", zig_toolchains = "toolchains")

# version, url_formats and host_platform_sha256 are optional, but highly
# recommended. Zig SDK is by default downloaded from dl.jakstys.lt, which is a
# tiny server in the closet of Yours Truly.
zig_toolchains(
    version = "0.10.0-dev.4253+fa9327ac0",
    url_formats = [
        "https://ziglang.org/builds/zig-{host_platform}-{version}.tar.xz",
    ],
    host_platform_sha256 = { "linux-x86_64": "a701d9f2f60d65feee20f60b8e729224187cab80805685ae207ab67dd13e8f1b" },
)