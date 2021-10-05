def cc_binary_rule(**kwargs):
    cuda_deps = ["@rules_cuda//cuda:cuda_runtime"]
    cuda_features = ["cuda", "-use_header_modules", "-layering_check", "-parse_headers"]
    srcs = kwargs.pop("srcs", [])
    srcs_gpu = kwargs.pop("srcs_gpu", [])
    srcs = srcs + srcs_gpu

    deps = kwargs.pop("deps", [])
    deps_gpu = kwargs.pop("deps_gpu", [])
    deps = select({":use_gpu":(deps_gpu + cuda_deps), 
                   ":use_mpi_gpu":(deps_gpu + cuda_deps), 
                   "//conditions:default":deps})

    features = kwargs.pop("features", [])
    features = select({":use_gpu":(features + cuda_features), 
                       ":use_mpi_gpu":(features + cuda_features), 
                       "//conditions:default":features})

    print("deps = " + repr(deps))
    print("features = " + repr(features))
    native.cc_binary(srcs = srcs, deps = deps, features=features, 
                     **kwargs)

