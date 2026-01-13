#!/bin/bash
# ==============================================================================
# tdLBCpp Test Script
# Simplified wrapper for running tests
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONFIG=""
VERBOSE=false
COVERAGE=false
TEST_FILTER=""

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [TEST_TARGET]

Run tests for tdLBCpp.

OPTIONS:
    -h, --help              Show this help message
    -c, --config CONFIG     Build configuration: cpu, gpu, debug, release
    -f, --filter PATTERN    Run only tests matching pattern
    --coverage              Generate coverage report
    -v, --verbose           Verbose test output

TEST_TARGET:
    Optional specific test target (default: //tdlbcpp/tests/...)

EXAMPLES:
    # Run all tests
    $0

    # Run specific test target
    $0 //tdlbcpp/tests/Params:tests

    # Run tests with verbose output
    $0 --verbose

    # Run tests matching a pattern
    $0 --filter "*Params*"

    # Run GPU tests
    $0 --config gpu

EOF
}

# Parse command line arguments
TEST_TARGET="//tdlbcpp/tests/..."
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
        -f|--filter)
            TEST_FILTER="$2"
            shift 2
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        //*)
            TEST_TARGET="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Build the bazel test command
BAZEL_CMD="bazel test"

# Add configuration
if [[ -n "$CONFIG" ]]; then
    BAZEL_CMD="$BAZEL_CMD --config=$CONFIG"
fi

# Add test filter
if [[ -n "$TEST_FILTER" ]]; then
    BAZEL_CMD="$BAZEL_CMD --test_filter=$TEST_FILTER"
fi

# Add coverage
if [[ "$COVERAGE" == true ]]; then
    BAZEL_CMD="$BAZEL_CMD --collect_code_coverage --combined_report=lcov"
fi

# Add verbose output
if [[ "$VERBOSE" == true ]]; then
    BAZEL_CMD="$BAZEL_CMD --test_output=all"
fi

# Add target
BAZEL_CMD="$BAZEL_CMD $TEST_TARGET"

# Print test information
print_info "Test Configuration:"
echo "  Config: ${CONFIG:-default}"
echo "  Target: $TEST_TARGET"
if [[ -n "$TEST_FILTER" ]]; then
    echo "  Filter: $TEST_FILTER"
fi
if [[ "$COVERAGE" == true ]]; then
    echo "  Coverage: enabled"
fi
echo ""

# Execute tests
print_info "Running tests..."
if [[ "$VERBOSE" == true ]]; then
    print_info "Command: $BAZEL_CMD"
fi

if eval $BAZEL_CMD; then
    print_success "All tests passed!"

    # Show coverage report if enabled
    if [[ "$COVERAGE" == true ]]; then
        print_info "Coverage report generated"
        print_info "View with: genhtml bazel-out/_coverage/_coverage_report.dat -o coverage_html"
    fi
else
    print_error "Some tests failed!"
    exit 1
fi
