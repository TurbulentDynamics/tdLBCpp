# Turbulent Dynamics Lattice Boltzmann (C++)

This is a basic version of the multi-node heterogeneous HPC code to run billions of cell simulation.




## Building
```
BAZEL_CXXOPTS="-std=c++14" bazel run --verbose_failures //tdlbcpp/src:tdlbcpp
```


```
WITH_GPU [1|0] depending on if GPU present
WTIH_GPU_SHARED [1|0] if useing shared Memory on GPU
```



