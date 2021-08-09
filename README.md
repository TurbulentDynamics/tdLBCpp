# Turbulent Dynamics Lattice Boltzmann (C++)

This is a basic version of the multi-node heterogeneous HPC code to run billions of cell simulation.




## Building
```
BAZEL_CXXOPTS="-std=c++14" bazel run --verbose_failures //tdlbcpp/src:tdlbcpp
```




