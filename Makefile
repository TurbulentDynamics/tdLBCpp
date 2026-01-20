# ==============================================================================
# tdLBCpp Makefile
# Convenient wrapper around Bazel build system
# ==============================================================================

.PHONY: all build test clean help setup run
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# Default configuration
CONFIG ?= cpu
BUILD_TYPE ?= release
CUDA_ARCH ?=
CUDA_PATH ?=

# Bazel targets
MAIN_TARGET := //tdlbcpp/src:tdlbcpp
TEST_TARGET := //tdlbcpp/tests/...
PARAMS_TEST := //tdlbcpp/tests/Params:tests

# Build binary path
BINARY := bazel-bin/tdlbcpp/src/tdlbcpp
SYMLINK := tdlbcpp-bin

# Build flags
BAZEL_BUILD_FLAGS := --config=$(BUILD_TYPE)
ifneq ($(CONFIG),)
	BAZEL_BUILD_FLAGS += --config=$(CONFIG)
endif
ifneq ($(CUDA_ARCH),)
	BAZEL_BUILD_FLAGS += --@rules_cuda//cuda:cuda_targets=$(CUDA_ARCH)
endif
ifneq ($(CUDA_PATH),)
	BAZEL_BUILD_FLAGS += --repo_env=CUDA_PATH=$(CUDA_PATH)
endif

# ==============================================================================
# Main Targets
# ==============================================================================

## Build the project (default: cpu release)
build:
	@echo "$(BLUE)Building tdLBCpp...$(NC)"
	@echo "Config: $(CONFIG), Type: $(BUILD_TYPE)"
	bazel build $(BAZEL_BUILD_FLAGS) $(MAIN_TARGET)
	@echo "$(GREEN)Build complete!$(NC)"
	@echo "Binary: $(BINARY)"
	@echo "$(BLUE)Creating symlink...$(NC)"
	@rm -f $(SYMLINK)
	@ln -s $(BINARY) $(SYMLINK)
	@echo "$(GREEN)Symlink created: $(SYMLINK) -> $(BINARY)$(NC)"

## Build everything (alias for build)
all: build

## Build CPU version (release)
cpu:
	@$(MAKE) build CONFIG=cpu BUILD_TYPE=release

## Build GPU version (release)
gpu:
	@$(MAKE) build CONFIG=gpu BUILD_TYPE=release

## Build GPU version with shared memory (release)
gpu-shared:
	@$(MAKE) build CONFIG=gpu_shared BUILD_TYPE=release

## Build MPI version (release)
mpi:
	@$(MAKE) build CONFIG=mpi BUILD_TYPE=release

## Build MPI + GPU version (release)
mpi-gpu:
	@$(MAKE) build CONFIG=mpi_gpu BUILD_TYPE=release

## Build Tegra version (release)
tegra:
	@$(MAKE) build CONFIG=tegra BUILD_TYPE=release

## Build debug version
debug:
	@$(MAKE) build BUILD_TYPE=debug

## Build CPU debug version
cpu-debug:
	@$(MAKE) build CONFIG=cpu BUILD_TYPE=debug

## Build GPU debug version
gpu-debug:
	@$(MAKE) build CONFIG=gpu BUILD_TYPE=debug

# ==============================================================================
# Testing
# ==============================================================================

## Run all tests
test:
	@echo "$(BLUE)Running tests...$(NC)"
	bazel test $(BAZEL_BUILD_FLAGS) $(TEST_TARGET)
	@echo "$(GREEN)All tests passed!$(NC)"

## Run tests with verbose output
test-verbose:
	@echo "$(BLUE)Running tests (verbose)...$(NC)"
	bazel test $(BAZEL_BUILD_FLAGS) --test_output=all $(TEST_TARGET)

## Run parameter tests only
test-params:
	@echo "$(BLUE)Running parameter tests...$(NC)"
	bazel test $(BAZEL_BUILD_FLAGS) $(PARAMS_TEST)

## Run tests with coverage
test-coverage:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	bazel test $(BAZEL_BUILD_FLAGS) --collect_code_coverage --combined_report=lcov $(TEST_TARGET)
	@echo "$(GREEN)Coverage report generated$(NC)"
	@echo "Generate HTML: genhtml bazel-out/_coverage/_coverage_report.dat -o coverage_html"

## Run GPU tests
test-gpu:
	@$(MAKE) test CONFIG=gpu

# ==============================================================================
# Debugging
# ==============================================================================

## Build with Address Sanitizer
asan:
	@echo "$(YELLOW)Building with Address Sanitizer...$(NC)"
	bazel build --config=asan $(MAIN_TARGET)
	@echo "$(GREEN)Build complete. Run to detect memory errors.$(NC)"

## Build with Thread Sanitizer
tsan:
	@echo "$(YELLOW)Building with Thread Sanitizer...$(NC)"
	bazel build --config=tsan $(MAIN_TARGET)
	@echo "$(GREEN)Build complete. Run to detect race conditions.$(NC)"

## Build with Undefined Behavior Sanitizer
ubsan:
	@echo "$(YELLOW)Building with UB Sanitizer...$(NC)"
	bazel build --config=ubsan $(MAIN_TARGET)
	@echo "$(GREEN)Build complete. Run to detect undefined behavior.$(NC)"

# ==============================================================================
# Running
# ==============================================================================

## Run the simulation (requires INPUT_FILE variable)
run: build
ifndef INPUT_FILE
	@echo "$(RED)ERROR: INPUT_FILE not specified$(NC)"
	@echo "Usage: make run INPUT_FILE=input.json"
	@exit 1
endif
	@echo "$(BLUE)Running simulation...$(NC)"
	@echo "Input file: $(INPUT_FILE)"
	$(BINARY) --input_file $(INPUT_FILE)

## Run with default test input
run-test: build
	@echo "$(BLUE)Running with test input (grid 60)...$(NC)"
	$(BINARY) --input_file input_debug_gridx60_numSteps20.json

## Generate input file (GRID_SIZE variable, default 100)
GRID_SIZE ?= 100
generate-input:
	@echo "$(BLUE)Generating input file (grid size: $(GRID_SIZE))...$(NC)"
	python3 generate_stirred_tank_input.py -x $(GRID_SIZE) -f input_grid$(GRID_SIZE).json
	@echo "$(GREEN)Generated: input_grid$(GRID_SIZE).json$(NC)"

## Generate and run simulation
run-generated:
	@$(MAKE) generate-input GRID_SIZE=$(GRID_SIZE)
	@$(MAKE) run INPUT_FILE=input_grid$(GRID_SIZE).json

# ==============================================================================
# Cleanup
# ==============================================================================

## Clean build artifacts
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	bazel clean
	@rm -f $(SYMLINK)
	@echo "$(GREEN)Clean complete$(NC)"

## Deep clean (removes all cached data)
clean-all:
	@echo "$(RED)Deep cleaning (removes all cached data)...$(NC)"
	bazel clean --expunge
	@rm -f $(SYMLINK)
	@echo "$(GREEN)Deep clean complete$(NC)"

## Clean generated files
clean-generated:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	rm -f input_grid*.json
	rm -rf coverage_html/
	@echo "$(GREEN)Generated files cleaned$(NC)"

# ==============================================================================
# Development
# ==============================================================================

## Setup development environment
setup:
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@chmod +x setup-dev.sh build.sh test.sh
	./setup-dev.sh

## Initialize git submodules
submodules:
	@echo "$(BLUE)Initializing git submodules...$(NC)"
	git submodule update --init --recursive
	@echo "$(GREEN)Submodules initialized$(NC)"

## Format code (if clang-format is available)
format:
	@if command -v clang-format >/dev/null 2>&1; then \
		echo "$(BLUE)Formatting code...$(NC)"; \
		find tdlbcpp -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' | xargs clang-format -i; \
		echo "$(GREEN)Formatting complete$(NC)"; \
	else \
		echo "$(YELLOW)clang-format not found, skipping$(NC)"; \
	fi

## Check build system
check:
	@echo "$(BLUE)Checking build system...$(NC)"
	@echo "Bazel version:"
	@bazel --version || echo "$(RED)Bazel not found$(NC)"
	@echo ""
	@echo "Python version:"
	@python3 --version || echo "$(RED)Python3 not found$(NC)"
	@echo ""
	@echo "C++ compiler:"
	@g++ --version 2>/dev/null || clang++ --version 2>/dev/null || echo "$(RED)No C++ compiler found$(NC)"
	@echo ""
	@echo "CUDA (optional):"
	@nvcc --version 2>/dev/null || echo "$(YELLOW)CUDA not found (optional)$(NC)"
	@echo ""
	@echo "MPI (optional):"
	@mpicc --version 2>/dev/null || echo "$(YELLOW)MPI not found (optional)$(NC)"

## Show build configuration
config:
	@echo "$(BLUE)Current Build Configuration:$(NC)"
	@echo "  CONFIG: $(CONFIG)"
	@echo "  BUILD_TYPE: $(BUILD_TYPE)"
	@echo "  CUDA_ARCH: $(CUDA_ARCH)"
	@echo "  CUDA_PATH: $(CUDA_PATH)"
	@echo "  BAZEL_FLAGS: $(BAZEL_BUILD_FLAGS)"

# ==============================================================================
# Quick Build + Test Combinations
# ==============================================================================

## Build and test CPU version
all-cpu: cpu test

## Build and test GPU version
all-gpu: gpu test-gpu

## Full pipeline: clean, build, test
full: clean build test
	@echo "$(GREEN)Full build and test complete!$(NC)"

## Quick development cycle: build and test
quick: build test
	@echo "$(GREEN)Quick build and test complete!$(NC)"

# ==============================================================================
# Code Formatting & Linting
# ==============================================================================

## Check code formatting (clang-format)
format-check:
	@echo "$(BLUE)Checking code formatting...$(NC)"
	@./format-code.sh --check
	@echo "$(GREEN)Format check complete!$(NC)"

## Fix code formatting (clang-format)
format:
	@echo "$(BLUE)Formatting code...$(NC)"
	@./format-code.sh --fix
	@echo "$(GREEN)Code formatted!$(NC)"

## Format specific file or directory
format-path:
	@echo "$(BLUE)Formatting $(PATH_TO_FORMAT)...$(NC)"
	@./format-code.sh --fix $(PATH_TO_FORMAT)

# ==============================================================================
# Examples
# ==============================================================================

## Run example workflow
example:
	@echo "$(BLUE)Running example workflow...$(NC)"
	@$(MAKE) generate-input GRID_SIZE=60
	@$(MAKE) cpu
	@$(MAKE) run INPUT_FILE=input_grid60.json
	@echo "$(GREEN)Example complete!$(NC)"

# ==============================================================================
# Help
# ==============================================================================

## Show this help message
help:
	@echo "$(GREEN)tdLBCpp Makefile - Available Targets$(NC)"
	@echo ""
	@echo "$(BLUE)Building:$(NC)"
	@echo "  make build          - Build with current config (default: cpu release)"
	@echo "  make cpu            - Build CPU version (release)"
	@echo "  make gpu            - Build GPU version (release)"
	@echo "  make gpu-shared     - Build GPU version with shared memory"
	@echo "  make mpi            - Build MPI version"
	@echo "  make mpi-gpu        - Build MPI + GPU version"
	@echo "  make debug          - Build debug version"
	@echo "  make gpu-debug      - Build GPU debug version"
	@echo ""
	@echo "$(BLUE)Testing:$(NC)"
	@echo "  make test           - Run all tests"
	@echo "  make test-verbose   - Run tests with verbose output"
	@echo "  make test-params    - Run parameter tests only"
	@echo "  make test-coverage  - Run tests with coverage"
	@echo "  make test-gpu       - Run GPU tests"
	@echo ""
	@echo "$(BLUE)Debugging:$(NC)"
	@echo "  make asan           - Build with Address Sanitizer"
	@echo "  make tsan           - Build with Thread Sanitizer"
	@echo "  make ubsan          - Build with UB Sanitizer"
	@echo ""
	@echo "$(BLUE)Running:$(NC)"
	@echo "  make run INPUT_FILE=file.json  - Run simulation"
	@echo "  make run-test                  - Run with test input"
	@echo "  make generate-input GRID_SIZE=100  - Generate input file"
	@echo "  make run-generated GRID_SIZE=100   - Generate and run"
	@echo ""
	@echo "$(BLUE)Cleanup:$(NC)"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make clean-all      - Deep clean (removes cache)"
	@echo "  make clean-generated - Clean generated files"
	@echo ""
	@echo "$(BLUE)Development:$(NC)"
	@echo "  make setup          - Setup development environment"
	@echo "  make submodules     - Initialize git submodules"
	@echo "  make format         - Format code with clang-format"
	@echo "  make check          - Check build system dependencies"
	@echo "  make config         - Show current build configuration"
	@echo ""
	@echo "$(BLUE)Combinations:$(NC)"
	@echo "  make all-cpu        - Build and test CPU version"
	@echo "  make all-gpu        - Build and test GPU version"
	@echo "  make full           - Clean, build, and test"
	@echo "  make quick          - Build and test (no clean)"
	@echo "  make example        - Run example workflow"
	@echo ""
	@echo "$(BLUE)Variables:$(NC)"
	@echo "  CONFIG=cpu|gpu|mpi|...  - Build configuration"
	@echo "  BUILD_TYPE=debug|release - Build type"
	@echo "  CUDA_ARCH=sm_75|sm_80|... - CUDA architecture"
	@echo "  CUDA_PATH=/path/to/cuda - CUDA installation path"
	@echo "  GRID_SIZE=100 - Grid size for input generation"
	@echo "  INPUT_FILE=file.json - Input file for running"
	@echo ""
	@echo "$(BLUE)Examples:$(NC)"
	@echo "  make gpu CUDA_ARCH=sm_75 CUDA_PATH=/usr/local/cuda-11.5"
	@echo "  make build CONFIG=mpi_gpu BUILD_TYPE=debug"
	@echo "  make run-generated GRID_SIZE=100"
	@echo "  make test CONFIG=gpu"
	@echo ""
	@echo "For detailed documentation, see BUILD_GUIDE.md"
