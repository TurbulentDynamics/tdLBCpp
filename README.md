# Turbulent Dynamics Lattice Boltzmann (C++)

This is a basic version of the multi-node heterogeneous HPC code to run billions of cell simulation.




## Building

### Generic build
```
bazel build //src:tdLBcpp --verbose_failures -s
```
### GPU build
uses `nvcc` compiler for cu files and main.cpp and sets WITH_GPU || WITH_GPU_MEMSHARED define
```
bazel build --config gpu //src:tdLBcpp
bazel build --config gpu_shared //src:tdLBcpp
```



