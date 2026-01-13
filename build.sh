#!/bin/bash
# ==============================================================================
# tdLBCpp Build Script
# Simplified wrapper around Bazel for common build configurations
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="release"
CONFIG=""
TARGET="//tdlbcpp/src:tdlbcpp"
CUDA_ARCH=""
CUDA_PATH=""
CLEAN=false
TEST=false
RUN=false
VERBOSE=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build tdLBCpp with various configurations.

OPTIONS:
    -h, --help              Show this help message
    -c, --config CONFIG     Build configuration: cpu, gpu, gpu_shared, mpi, mpi_gpu, tegra
    -t, --type TYPE         Build type: debug, release (default: release)
    -a, --arch ARCH         CUDA architecture (e.g., sm_75, sm_80, sm_86)
    -p, --cuda-path PATH    Path to CUDA installation
    --clean                 Clean build artifacts before building
    --test                  Run tests after building
    --run                   Run the binary after building
    -v, --verbose           Verbose output

EXAMPLES:
    # Build CPU version in release mode
    $0 --config cpu --type release

    # Build GPU version with specific CUDA architecture
    $0 --config gpu --arch sm_75 --cuda-path /usr/local/cuda-11.5

    # Build and test
    $0 --config gpu --test

    # Clean build of MPI+GPU version
    $0 --clean --config mpi_gpu --type debug

    # Build, test, and run
    $0 --config cpu --test --run

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -a|--arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        -p|--cuda-path)
            CUDA_PATH="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --test)
            TEST=true
            shift
            ;;
        --run)
            RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate build type
if [[ "$BUILD_TYPE" != "debug" && "$BUILD_TYPE" != "release" ]]; then
    print_error "Invalid build type: $BUILD_TYPE (must be 'debug' or 'release')"
    exit 1
fi

# Build the bazel command
BAZEL_CMD="bazel build"

# Add build type
BAZEL_CMD="$BAZEL_CMD --config=$BUILD_TYPE"

# Add configuration
if [[ -n "$CONFIG" ]]; then
    BAZEL_CMD="$BAZEL_CMD --config=$CONFIG"
fi

# Add CUDA architecture if specified
if [[ -n "$CUDA_ARCH" ]]; then
    BAZEL_CMD="$BAZEL_CMD --@rules_cuda//cuda:cuda_targets=$CUDA_ARCH"
fi

# Add CUDA path if specified
if [[ -n "$CUDA_PATH" ]]; then
    BAZEL_CMD="$BAZEL_CMD --repo_env=CUDA_PATH=$CUDA_PATH"
fi

# Add target
BAZEL_CMD="$BAZEL_CMD $TARGET"

# Clean if requested
if [[ "$CLEAN" == true ]]; then
    print_info "Cleaning build artifacts..."
    bazel clean
    print_success "Clean complete"
fi

# Print build information
print_info "Build Configuration:"
echo "  Build Type: $BUILD_TYPE"
echo "  Config: ${CONFIG:-default}"
echo "  Target: $TARGET"
if [[ -n "$CUDA_ARCH" ]]; then
    echo "  CUDA Architecture: $CUDA_ARCH"
fi
if [[ -n "$CUDA_PATH" ]]; then
    echo "  CUDA Path: $CUDA_PATH"
fi
echo ""

# Execute build
print_info "Building tdLBCpp..."
if [[ "$VERBOSE" == true ]]; then
    print_info "Command: $BAZEL_CMD"
fi

if eval $BAZEL_CMD; then
    print_success "Build completed successfully!"

    # Find the built binary
    BINARY_PATH=$(bazel cquery --output=files $TARGET 2>/dev/null | head -n1)
    if [[ -n "$BINARY_PATH" ]]; then
        print_info "Binary location: $BINARY_PATH"
    fi
else
    print_error "Build failed!"
    exit 1
fi

# Run tests if requested
if [[ "$TEST" == true ]]; then
    print_info "Running tests..."
    if bazel test //tdlbcpp/tests/...; then
        print_success "All tests passed!"
    else
        print_error "Some tests failed!"
        exit 1
    fi
fi

# Run the binary if requested
if [[ "$RUN" == true ]]; then
    if [[ -n "$BINARY_PATH" && -f "$BINARY_PATH" ]]; then
        print_info "Running tdLBCpp..."
        echo ""
        "$BINARY_PATH" --help || true
    else
        print_warning "Could not find binary to run"
    fi
fi

print_success "All done!"
