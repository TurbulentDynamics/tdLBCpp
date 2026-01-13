# tdLBCpp Quick Reference

Fast reference for common commands and workflows.

## Quick Start (First Time)

```bash
git clone --recursive https://github.com/TurbulentDynamics/tdLBCpp
cd tdLBCpp
make setup
```

## Most Common Commands

```bash
make help              # Show all available commands
make cpu               # Build CPU version
make gpu               # Build GPU version
make test              # Run tests
make run-generated     # Generate input and run
```

## Build Commands

| Command | Description |
|---------|-------------|
| `make cpu` | Build CPU version (release) |
| `make gpu` | Build GPU version (release) |
| `make gpu-shared` | Build GPU with shared memory |
| `make mpi` | Build MPI version |
| `make mpi-gpu` | Build MPI + GPU version |
| `make debug` | Build debug version |
| `make gpu-debug` | Build GPU debug version |

## Test Commands

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make test-verbose` | Run tests with verbose output |
| `make test-params` | Run parameter tests only |
| `make test-coverage` | Generate coverage report |
| `make test-gpu` | Run GPU tests |

## Run Commands

| Command | Description |
|---------|-------------|
| `make generate-input GRID_SIZE=100` | Generate input file |
| `make run INPUT_FILE=input.json` | Run simulation |
| `make run-generated GRID_SIZE=100` | Generate and run |
| `make run-test` | Run with test input |

## Debug Commands

| Command | Description |
|---------|-------------|
| `make asan` | Build with Address Sanitizer |
| `make tsan` | Build with Thread Sanitizer |
| `make ubsan` | Build with UB Sanitizer |

## Clean Commands

| Command | Description |
|---------|-------------|
| `make clean` | Clean build artifacts |
| `make clean-all` | Deep clean (removes cache) |
| `make clean-generated` | Clean generated files |

## Development Commands

| Command | Description |
|---------|-------------|
| `make setup` | Setup dev environment |
| `make check` | Check dependencies |
| `make format` | Format code |
| `make config` | Show current config |

## Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CONFIG` | Build configuration | `make build CONFIG=gpu` |
| `BUILD_TYPE` | Build type | `make build BUILD_TYPE=debug` |
| `CUDA_ARCH` | CUDA architecture | `make gpu CUDA_ARCH=sm_75` |
| `CUDA_PATH` | CUDA installation path | `make gpu CUDA_PATH=/usr/local/cuda-11.5` |
| `GRID_SIZE` | Grid size for input | `make generate-input GRID_SIZE=200` |
| `INPUT_FILE` | Input file to run | `make run INPUT_FILE=input.json` |

## Common Workflows

### CPU Development
```bash
make all-cpu                      # Build and test
make run-generated GRID_SIZE=60   # Run simulation
```

### GPU Development
```bash
make gpu CUDA_ARCH=sm_75 CUDA_PATH=/usr/local/cuda-11.5
make test-gpu
make run-generated GRID_SIZE=100
```

### Debug Memory Issues
```bash
make asan
make run INPUT_FILE=input.json
```

### Full Clean Build
```bash
make clean
make cpu
make test
```

## CUDA Architectures

| GPU | Architecture | Example |
|-----|--------------|---------|
| GTX 1080 | Pascal | `sm_61` |
| RTX 2080 | Turing | `sm_75` |
| A100 | Ampere | `sm_80` |
| RTX 3090 | Ampere | `sm_86` |
| H100 | Hopper | `sm_90` |

## File Locations

| Item | Location |
|------|----------|
| Binary | `bazel-bin/tdlbcpp/src/tdlbcpp` |
| Build config | `.bazelrc` |
| User config | `.bazelrc.user` |
| Build files | `tdlbcpp/src/BUILD` |
| Tests | `tdlbcpp/tests/` |

## Alternative Build Methods

### Using Shell Scripts
```bash
./build.sh --config gpu --arch sm_75
./test.sh --verbose
```

### Using Bazel Directly
```bash
bazel build --config=gpu //tdlbcpp/src:tdlbcpp
bazel test //tdlbcpp/tests/...
```

## Getting Help

```bash
make help           # Show Makefile help
./build.sh --help   # Show build script help
./test.sh --help    # Show test script help
```

## Documentation

- [BUILD_GUIDE.md](BUILD_GUIDE.md) - Complete build documentation
- [README.md](README.md) - Project overview
- `make help` - Command reference

## Troubleshooting

### Build Fails
```bash
make check          # Check dependencies
make clean          # Clean and retry
make clean-all      # Deep clean
```

### CUDA Not Found
```bash
# Set CUDA path explicitly
make gpu CUDA_PATH=/usr/local/cuda-11.5

# Or add to .bazelrc.user:
echo 'build:gpu --repo_env=CUDA_PATH=/usr/local/cuda-11.5' >> .bazelrc.user
```

### Out of Memory
```bash
# Reduce parallel jobs in .bazelrc.user:
echo 'build --jobs=4' >> .bazelrc.user
```

### Submodules Missing
```bash
make submodules
```
