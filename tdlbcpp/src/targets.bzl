load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

def cc_binary_rule(**kwargs):
    cuda_deps = ["@rules_cuda//cuda:cuda_runtime"]
    cuda_features = ["cuda", "-use_header_modules", "-layering_check", "-parse_headers"]
    srcs = kwargs.pop("srcs", [])
    srcs_gpu = kwargs.pop("srcs_gpu", [])
    srcs = srcs + srcs_gpu

    deps = kwargs.pop("deps", [])
    deps_gpu = kwargs.pop("deps_gpu", [])
    deps_zig = kwargs.pop("deps_zig", [])
    deps = select({":use_gpu":(deps_gpu + cuda_deps), 
                   ":use_mpi_gpu":(deps_gpu + cuda_deps), 
                   ":use_zig":(deps_zig), 
                   "//conditions:default":deps})

    features = kwargs.pop("features", [])
    features = select({":use_gpu":(features + cuda_features), 
                       ":use_mpi_gpu":(features + cuda_features), 
                       "//conditions:default":features})

    print("deps = " + repr(deps))
    print("features = " + repr(features))
    native.cc_binary(srcs = srcs, deps = deps, features=features, 
                     **kwargs)

def _zig_rule(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    output_file = ctx.actions.declare_file(ctx.label.name + ".a")

    srcs = ctx.files.srcs
    hdrs = ctx.files.hdrs
    paths = [src.path for src in srcs]
    copts = ctx.attr.copts
    includes_folders = list([k for k in dict([(hdr.dirname, 1) for hdr in hdrs if hdr.path.lower().endswith(".h")]).keys()])
    includes_opts = [("-I" + item) for item in includes_folders]

    ctx.actions.run(
        mnemonic = "zig",
        executable = ctx.attr.compiler.files_to_run.executable,
        arguments = copts + ["-femit-bin=" + output_file.path] + includes_opts + paths,
        inputs = depset(srcs + [ctx.attr.compiler.files_to_run.executable]),
        tools = ctx.files.compiler_data + hdrs,
        outputs = [output_file],
    )

    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(direct = [
            cc_common.create_library_to_link(
                actions = ctx.actions,
                feature_configuration = feature_configuration,
                cc_toolchain = cc_toolchain,
                static_library = output_file,
            ),
        ]),
    )

    linking_context = cc_common.create_linking_context(linker_inputs = depset(direct = [linker_input]))

    (compilation_context, compilation_outputs) = cc_common.compile(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = ctx.files.srcs,
        public_hdrs = ctx.files.hdrs,
        user_compile_flags = ctx.attr.copts,
    )

    return [CcInfo(linking_context = linking_context, compilation_context = compilation_context)]

zig_lib = rule(
    implementation = _zig_rule,
    executable=False,
    attrs = {
        "srcs": attr.label_list(allow_files = [".zig", ".h"]),
        "hdrs": attr.label_list(allow_files = [".zig", ".h"]),
        "copts": attr.string_list(),
        "compiler": attr.label(
            default = Label("@zig_sdk//:tools/build-lib"),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "compiler_data": attr.label(
            default = Label("@zig_sdk//:all"),
            providers = ["files"],
        ),
        "build_command": attr.string(default = "build-lib"),
        "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain()
)