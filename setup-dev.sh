#!/bin/bash
# ==============================================================================
# tdLBCpp Development Environment Setup Script
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

print_header() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

# Check for required tools
check_dependencies() {
    print_header "Checking Dependencies"

    local missing_deps=()

    # Check for Bazel
    if command -v bazel &> /dev/null; then
        BAZEL_VERSION=$(bazel --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
        print_success "Bazel found (version $BAZEL_VERSION)"
    else
        print_warning "Bazel not found"
        missing_deps+=("bazel")
    fi

    # Check for Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        print_success "Python3 found (version $PYTHON_VERSION)"
    else
        print_warning "Python3 not found"
        missing_deps+=("python3")
    fi

    # Check for Git
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        print_success "Git found (version $GIT_VERSION)"
    else
        print_warning "Git not found"
        missing_deps+=("git")
    fi

    # Check for C++ compiler
    if command -v g++ &> /dev/null; then
        GCC_VERSION=$(g++ --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
        print_success "g++ found (version $GCC_VERSION)"
    elif command -v clang++ &> /dev/null; then
        CLANG_VERSION=$(clang++ --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
        print_success "clang++ found (version $CLANG_VERSION)"
    else
        print_warning "No C++ compiler found"
        missing_deps+=("g++ or clang++")
    fi

    # Check for CUDA (optional)
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
        print_success "CUDA found (version $CUDA_VERSION)"
        print_info "CUDA path: $(which nvcc | sed 's|/bin/nvcc||')"
    else
        print_info "CUDA not found (optional, required for GPU builds)"
    fi

    # Check for MPI (optional)
    if command -v mpicc &> /dev/null; then
        print_success "MPI compiler found"
    else
        print_info "MPI not found (optional, required for MPI builds)"
    fi

    # Report missing dependencies
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo ""
        print_error "Missing required dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        print_info "Install instructions:"
        echo "  macOS:   brew install bazel python3"
        echo "  Ubuntu:  sudo apt-get install bazel python3 g++"
        echo ""
        return 1
    fi

    print_success "All required dependencies found!"
    return 0
}

# Initialize git submodules
init_submodules() {
    print_header "Initializing Git Submodules"

    if [ -d ".git" ]; then
        git submodule update --init --recursive
        print_success "Git submodules initialized"
    else
        print_warning "Not a git repository"
    fi
}

# Verify build system
verify_build() {
    print_header "Verifying Build System"

    # Check if WORKSPACE exists
    if [ ! -f "WORKSPACE" ]; then
        print_error "WORKSPACE file not found"
        return 1
    fi

    print_info "Testing basic build..."
    if bazel build //tdlbcpp/src:computeUnit &> /dev/null; then
        print_success "Build system verified"
    else
        print_warning "Build test failed (this may be normal if dependencies are not set up)"
    fi
}

# Create user bazelrc for local overrides
create_user_bazelrc() {
    print_header "Creating User Configuration"

    if [ ! -f ".bazelrc.user" ]; then
        cat > .bazelrc.user << EOF
# Local user-specific Bazel configuration
# This file is ignored by git and can contain machine-specific settings

# Example: Set custom CUDA path
# build:gpu --repo_env=CUDA_PATH=/usr/local/cuda-12.0

# Example: Set custom number of jobs
# build --jobs=8

# Example: Enable disk cache
# build --disk_cache=~/.cache/bazel

EOF
        print_success "Created .bazelrc.user template"
        print_info "Edit .bazelrc.user to add local configurations"
    else
        print_info ".bazelrc.user already exists"
    fi
}

# Print environment info
print_env_info() {
    print_header "Environment Information"

    echo "Operating System: $(uname -s)"
    echo "Architecture: $(uname -m)"
    echo "Working Directory: $(pwd)"

    if [ -n "$CUDA_VERSION" ]; then
        echo "CUDA Support: Yes (v$CUDA_VERSION)"
    else
        echo "CUDA Support: No"
    fi

    if command -v mpicc &> /dev/null; then
        echo "MPI Support: Yes"
    else
        echo "MPI Support: No"
    fi
}

# Print next steps
print_next_steps() {
    print_header "Setup Complete!"

    cat << EOF
Next steps:

1. Build the project:
   ./build.sh --config cpu --type release

2. For GPU builds (if CUDA is available):
   ./build.sh --config gpu --arch sm_75 --cuda-path /usr/local/cuda-11.5

3. Run tests:
   ./test.sh

4. Generate input files and run simulation:
   python3 generate_stirred_tank_input.py -x 100 -f input.json
   bazel-bin/tdlbcpp/src/tdlbcpp --input_file input.json

5. For more options:
   ./build.sh --help
   ./test.sh --help

For documentation, see README.md and docs/

EOF

    print_success "Development environment is ready!"
}

# Main setup flow
main() {
    print_header "tdLBCpp Development Environment Setup"

    check_dependencies || exit 1
    init_submodules
    verify_build
    create_user_bazelrc
    print_env_info
    print_next_steps
}

# Run main
main
