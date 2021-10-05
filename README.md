# Turbulent Dynamics Lattice Boltzmann (C++)

This is a basic version of the multi-node heterogeneous HPC code to run simulation with hundreds of billions of cells.





## Building

### Generic build
```
bazel build //tdlbcpp/src:tdlbcpp --verbose_failures -s
```
### GPU build
uses `nvcc` compiler for cuda files and main.cpp and sets WITH_GPU | WITH_GPU_MEMSHARED define
```
bazel build --config gpu //tdlbcpp/src:tdlbcpp
bazel build --config gpu_shared //tdlbcpp/src:tdlbcpp
```

For debug build add -c dbg switch:
```
bazel build -c dbg --config gpu //tdlbcpp/src:tdlbcpp
```

Specify capabilities and custom cuda path using the following switches:

`--@rules_cuda//cuda:cuda_targets=sm_XY`
`--repo_env=CUDA_PATH=/usr/local/cuda-ZZZ`

```
bazel build -c dbg --config gpu --@rules_cuda//cuda:cuda_targets=sm_30 --repo_env=CUDA_PATH=/usr/local/cuda-9.1 //tdlbcpp/src:tdlbcpp
```

## Package Structure
![Package Structure](docs/Package-Structure.jpeg)




## Vector Identification (see [Header.h](tdlbcpp/src/Header.h))
![D2Q9 and D3Q19](docs/D2Q9-D3Q19.jpeg)

