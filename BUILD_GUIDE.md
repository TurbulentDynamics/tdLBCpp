# tdLBCpp Build Guide

Complete guide for building and developing tdLBCpp - Turbulent Dynamics Lattice Boltzmann C++ code.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dependencies](#dependencies)
3. [Build Configurations](#build-configurations)
4. [Building with Make](#building-with-make)
5. [Build Scripts](#build-scripts)
6. [Testing](#testing)
7. [Advanced Options](#advanced-options)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Initial Setup

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/TurbulentDynamics/tdLBCpp

# Run the development environment setup
cd tdLBCpp
./setup-dev.sh
```

### Basic Build

```bash
# Build CPU version (release mode)
./build.sh --config cpu --type release

# Build and run tests
./build.sh --config cpu --type release --test
```

### Run Simulation

```bash
# Generate input file
python3 generate_stirred_tank_input.py -x 100 -f input.json

# Run simulation
bazel-bin/tdlbcpp/src/tdlbcpp --input_file input.json
```

---

## Dependencies

### Required Dependencies

- **Bazel** (build system) - version 5.0 or later
- **Python 3** - for input file generation
- **C++ Compiler** - g++ 7.4+ or clang++ with C++17 support
- **Git** - for version control and submodules

### Optional Dependencies

- **CUDA Toolkit** - for GPU acceleration (CUDA 11.0+)
- **MPI** - for distributed computing (OpenMPI or MPICH)
- **Spack** - alternative package manager (see `spack.yaml`)

### Installing Dependencies

**macOS:**
```bash
brew install bazel python3
# For CUDA support, install CUDA Toolkit from NVIDIA
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install bazel python3 g++ git

# For GPU support
sudo apt-get install nvidia-cuda-toolkit

# For MPI support
sudo apt-get install libopenmpi-dev openmpi-bin
```

**Using Spack:**
```bash
# Install Spack
git clone https://github.com/spack/spack.git ~/spack
. ~/spack/share/spack/setup-env.sh

# Create and activate environment
spack env create tdlb spack.yaml
spack env activate -p tdlb
spack install
```

---

## Building with Make

The Makefile provides the simplest and most convenient way to build tdLBCpp. It wraps the Bazel build system with easy-to-use targets.

### Quick Start with Make

```bash
# Show all available targets
make help

# Build CPU version (default)
make cpu

# Build GPU version
make gpu

# Build and test
make all-cpu

# Clean and rebuild
make clean build
```

### Common Make Targets

**Building:**
```bash
make cpu           # Build CPU version (release)
make gpu           # Build GPU version (release)
make gpu-shared    # Build GPU with shared memory
make mpi           # Build MPI version
make mpi-gpu       # Build MPI + GPU version
make debug         # Build debug version
make cpu-debug     # Build CPU debug version
make gpu-debug     # Build GPU debug version
```

**Testing:**
```bash
make test          # Run all tests
make test-verbose  # Run tests with verbose output
make test-params   # Run parameter tests only
make test-coverage # Run tests with coverage
make test-gpu      # Run GPU tests
```

**Running:**
```bash
# Generate input file
make generate-input GRID_SIZE=100

# Run simulation
make run INPUT_FILE=input.json

# Generate and run in one step
make run-generated GRID_SIZE=100

# Run with test input
make run-test
```

**Debugging:**
```bash
make asan          # Build with Address Sanitizer
make tsan          # Build with Thread Sanitizer
make ubsan         # Build with UB Sanitizer
```

**Cleanup:**
```bash
make clean         # Clean build artifacts
make clean-all     # Deep clean (removes cache)
make clean-generated  # Clean generated files
```

**Development:**
```bash
make setup         # Setup development environment
make submodules    # Initialize git submodules
make format        # Format code with clang-format
make check         # Check build dependencies
make config        # Show current configuration
```

**Combinations:**
```bash
make all-cpu       # Build and test CPU version
make all-gpu       # Build and test GPU version
make full          # Clean, build, and test
make quick         # Build and test (no clean)
make example       # Run example workflow
```

### Make Variables

Customize builds using variables:

```bash
# Set build configuration
make build CONFIG=gpu BUILD_TYPE=debug

# Set CUDA architecture
make gpu CUDA_ARCH=sm_75

# Set CUDA path
make gpu CUDA_PATH=/usr/local/cuda-11.5

# Combine multiple variables
make gpu CUDA_ARCH=sm_86 CUDA_PATH=/opt/cuda-12.0 BUILD_TYPE=release

# Generate input with specific grid size
make generate-input GRID_SIZE=200

# Run with specific input file
make run INPUT_FILE=my_input.json
```

**Available Variables:**
- `CONFIG` - Build configuration (cpu, gpu, mpi, mpi_gpu, gpu_shared, tegra)
- `BUILD_TYPE` - Build type (debug, release)
- `CUDA_ARCH` - CUDA architecture (sm_75, sm_80, sm_86, etc.)
- `CUDA_PATH` - Path to CUDA installation
- `GRID_SIZE` - Grid size for input generation (default: 100)
- `INPUT_FILE` - Input file for running simulation

### Complete Workflow Examples

**CPU Development Workflow:**
```bash
# Setup environment (first time only)
make setup

# Build and test
make all-cpu

# Generate input and run
make run-generated GRID_SIZE=60
```

**GPU Development Workflow:**
```bash
# Build GPU version with specific architecture
make gpu CUDA_ARCH=sm_75 CUDA_PATH=/usr/local/cuda-11.5

# Run tests
make test-gpu

# Generate and run simulation
make generate-input GRID_SIZE=100
make run INPUT_FILE=input_grid100.json
```

**Debug Workflow:**
```bash
# Build with Address Sanitizer
make asan

# Run to detect memory issues
make run INPUT_FILE=input_grid60.json

# If issues found, build debug version
make gpu-debug

# Run with debugger
gdb bazel-bin/tdlbcpp/src/tdlbcpp
```

**Complete Build and Test:**
```bash
# Full clean build and test
make full

# Or quick iteration without clean
make quick
```

---

## Build Configurations

### Configuration Profiles

tdLBCpp supports multiple build configurations defined in `.bazelrc`:

#### Compute Configurations

| Config | Description | Use Case |
|--------|-------------|----------|
| `cpu` | CPU-only build | Default, no special hardware |
| `gpu` | GPU-accelerated (CUDA) | Single GPU workstation |
| `gpu_shared` | GPU with shared memory | GPU with shared memory support |
| `mpi` | MPI distributed computing | Multi-node clusters |
| `mpi_gpu` | MPI + GPU | GPU clusters |
| `tegra` | NVIDIA Tegra platforms | Embedded GPU systems |

#### Build Types

| Type | Description | Optimization | Debug Info |
|------|-------------|--------------|------------|
| `debug` | Debug build | `-O0` | Full symbols |
| `release` | Optimized build | `-O3` | Stripped |

#### Special Configurations

| Config | Description | Use Case |
|--------|-------------|----------|
| `asan` | Address Sanitizer | Memory error detection |
| `tsan` | Thread Sanitizer | Race condition detection |
| `ubsan` | UB Sanitizer | Undefined behavior detection |

---

## Build Scripts

### build.sh - Main Build Script

Simplified wrapper for building tdLBCpp.

**Basic Usage:**
```bash
./build.sh [OPTIONS]
```

**Common Examples:**

```bash
# CPU release build
./build.sh --config cpu --type release

# GPU debug build
./build.sh --config gpu --type debug

# GPU build with specific architecture
./build.sh --config gpu --arch sm_75 --cuda-path /usr/local/cuda-11.5

# Clean build
./build.sh --clean --config gpu

# Build and test
./build.sh --config cpu --test

# Build, test, and run
./build.sh --config cpu --test --run
```

**Options:**

- `-c, --config CONFIG` - Build configuration (cpu, gpu, mpi, etc.)
- `-t, --type TYPE` - Build type (debug, release)
- `-a, --arch ARCH` - CUDA architecture (e.g., sm_75, sm_80, sm_86)
- `-p, --cuda-path PATH` - Path to CUDA installation
- `--clean` - Clean before building
- `--test` - Run tests after building
- `--run` - Run the binary after building
- `-v, --verbose` - Verbose output
- `-h, --help` - Show help

### Direct Bazel Commands

For more control, use Bazel directly:

```bash
# Generic CPU build
bazel build //tdlbcpp/src:tdlbcpp

# GPU build
bazel build --config=gpu //tdlbcpp/src:tdlbcpp

# GPU build with specific settings
bazel build --config=gpu \
  --@rules_cuda//cuda:cuda_targets=sm_75 \
  --repo_env=CUDA_PATH=/usr/local/cuda-11.5 \
  //tdlbcpp/src:tdlbcpp

# Debug GPU build
bazel build -c dbg --config=gpu //tdlbcpp/src:tdlbcpp

# MPI + GPU build
bazel build --config=mpi_gpu //tdlbcpp/src:tdlbcpp
```

---

## Testing

### test.sh - Test Script

Simplified wrapper for running tests.

**Basic Usage:**
```bash
./test.sh [OPTIONS] [TEST_TARGET]
```

**Examples:**

```bash
# Run all tests
./test.sh

# Run specific test target
./test.sh //tdlbcpp/tests/Params:tests

# Run with verbose output
./test.sh --verbose

# Run tests matching pattern
./test.sh --filter "*Params*"

# Run GPU tests
./test.sh --config gpu

# Generate coverage report
./test.sh --coverage
```

**Options:**

- `-c, --config CONFIG` - Build configuration
- `-f, --filter PATTERN` - Run only tests matching pattern
- `--coverage` - Generate coverage report
- `-v, --verbose` - Verbose test output
- `-h, --help` - Show help

### Direct Bazel Test Commands

```bash
# Run all tests
bazel test //tdlbcpp/tests/...

# Run specific test
bazel test //tdlbcpp/tests/Params:tests

# Run tests with detailed output
bazel test --test_output=all //tdlbcpp/tests/...

# Run tests with coverage
bazel test --collect_code_coverage //tdlbcpp/tests/...
```

---

## Advanced Options

### CUDA Architecture Selection

Specify the GPU compute capability for optimal performance:

```bash
# For RTX 2080 Ti (Turing, sm_75)
./build.sh --config gpu --arch sm_75

# For RTX 3090 (Ampere, sm_86)
./build.sh --config gpu --arch sm_86

# For RTX 4090 (Ada, sm_89)
./build.sh --config gpu --arch sm_89

# For A100 (sm_80)
./build.sh --config gpu --arch sm_80
```

**Common Compute Capabilities:**
- `sm_60`, `sm_61` - Pascal (P100, GTX 1080)
- `sm_70`, `sm_75` - Volta/Turing (V100, RTX 2080)
- `sm_80`, `sm_86` - Ampere (A100, RTX 3090)
- `sm_89`, `sm_90` - Ada/Hopper (RTX 4090, H100)

### Custom CUDA Installation Path

```bash
./build.sh --config gpu \
  --cuda-path /usr/local/cuda-11.8 \
  --arch sm_86
```

### User-Specific Configuration

Create `.bazelrc.user` for local machine-specific settings:

```bash
# .bazelrc.user example
build:gpu --repo_env=CUDA_PATH=/opt/cuda-12.0
build --jobs=16
build --disk_cache=~/.cache/bazel
```

This file is ignored by git and won't interfere with the repository.

### Performance Tuning

```bash
# Use all CPU cores
bazel build --jobs=auto //tdlbcpp/src:tdlbcpp

# Limit parallel jobs
bazel build --jobs=8 //tdlbcpp/src:tdlbcpp

# Enable disk cache
bazel build --disk_cache=~/.cache/bazel //tdlbcpp/src:tdlbcpp

# Remote caching (if available)
bazel build --remote_cache=grpc://cache-server:9092 //tdlbcpp/src:tdlbcpp
```

### Debugging Tools

**Address Sanitizer (memory errors):**
```bash
bazel build --config=asan //tdlbcpp/src:tdlbcpp
bazel-bin/tdlbcpp/src/tdlbcpp --input_file input.json
```

**Thread Sanitizer (race conditions):**
```bash
bazel build --config=tsan //tdlbcpp/src:tdlbcpp
bazel-bin/tdlbcpp/src/tdlbcpp --input_file input.json
```

**Undefined Behavior Sanitizer:**
```bash
bazel build --config=ubsan //tdlbcpp/src:tdlbcpp
bazel-bin/tdlbcpp/src/tdlbcpp --input_file input.json
```

---

## Troubleshooting

### Common Issues

#### Bazel Not Found

```bash
# Install Bazel
# macOS:
brew install bazel

# Ubuntu:
sudo apt install bazel
```

#### CUDA Not Found

```bash
# Specify CUDA path explicitly
./build.sh --config gpu --cuda-path /usr/local/cuda-11.5

# Or add to .bazelrc.user:
echo 'build:gpu --repo_env=CUDA_PATH=/usr/local/cuda-11.5' >> .bazelrc.user
```

#### Build Fails with "Cannot find cuda_samples"

The project requires CUDA helper headers:

```bash
# Download CUDA samples
git clone https://github.com/NVIDIA/cuda-samples.git /tmp/cuda-samples

# Copy helper headers
cp /tmp/cuda-samples/Common/{helper_*.h,exception.h} tdlbcpp/src/
```

#### MPI Compiler Not Found

```bash
# Ubuntu:
sudo apt-get install libopenmpi-dev openmpi-bin

# macOS:
brew install open-mpi
```

#### Submodule Not Initialized

```bash
git submodule update --init --recursive
```

#### Out of Memory During Build

```bash
# Reduce parallel jobs
bazel build --jobs=4 //tdlbcpp/src:tdlbcpp

# Or set in .bazelrc.user:
echo 'build --jobs=4' >> .bazelrc.user
```

### Clean Build

If you encounter persistent issues:

```bash
# Clean all build artifacts
bazel clean

# Deep clean (removes all cached data)
bazel clean --expunge

# Rebuild
./build.sh --config cpu --type release
```

### Getting Help

- Check existing issues: https://github.com/TurbulentDynamics/tdLBCpp/issues
- Review package structure: `docs/Package-Structure.jpeg`
- Examine build configuration: `.bazelrc`, `WORKSPACE`, `BUILD` files

---

## Migration Notes

### From Old Build System

If you're familiar with the previous build process:

**Old:**
```bash
bazel build //tdlbcpp/src:tdlbcpp --verbose_failures -s
```

**New (equivalent):**
```bash
./build.sh --config cpu --type release --verbose
```

**Old GPU build:**
```bash
bazel build --config gpu --@rules_cuda//cuda:cuda_targets=sm_75 \
  --repo_env=CUDA_PATH=/usr/local/cuda-11.5 //tdlbcpp/src:tdlbcpp
```

**New (equivalent):**
```bash
./build.sh --config gpu --arch sm_75 --cuda-path /usr/local/cuda-11.5
```

### New Features

- ✅ Simplified build scripts (`build.sh`, `test.sh`, `setup-dev.sh`)
- ✅ Updated GoogleTest (v1.15.2)
- ✅ Enhanced `.bazelrc` with multiple profiles
- ✅ MODULE.bazel for Bazel 7+ support
- ✅ Sanitizer configurations (asan, tsan, ubsan)
- ✅ User-specific configuration (`.bazelrc.user`)
- ✅ Platform-specific optimizations
- ✅ Improved test runner with coverage support

---

## Additional Resources

- [README.md](README.md) - Project overview
- [docs/](docs/) - Additional documentation
- [Bazel Documentation](https://bazel.build/docs)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [MPI Documentation](https://www.open-mpi.org/doc/)
