# Turbulent Dynamics Lattice Boltzmann (C++)

This is a basic version of the multi-node heterogeneous HPC code to run simulation with hundreds of billions of cells.





## Building

### Generic build
```
bazel build //src:tdLBcpp --verbose_failures -s
```
### GPU build
uses `nvcc` compiler for cuda files and main.cpp and sets WITH_GPU | WITH_GPU_MEMSHARED define
```
bazel build --config gpu //src:tdLBcpp
bazel build --config gpu_shared //src:tdLBcpp
```


## Package Structure
![Package Structure](docs/Package-Structure.jpeg)




## Vector Identification (see [Header.h](tdlbcpp/src/Header.h))
![D2Q9 and D3Q19](docs/D2Q9-D3Q19.jpeg)

