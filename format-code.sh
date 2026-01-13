#!/bin/bash
# Code formatting script for tdLBCpp
# Usage: ./format-code.sh [--check|--fix] [path]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODE="check"
TARGET_PATH="."

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            MODE="check"
            shift
            ;;
        --fix)
            MODE="fix"
            shift
            ;;
        *)
            TARGET_PATH="$1"
            shift
            ;;
    esac
done

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
    echo -e "${RED}Error: clang-format is not installed${NC}"
    echo "Install it with: brew install clang-format"
    exit 1
fi

echo "clang-format version: $(clang-format --version)"
echo ""

# Find all C++ files
CPP_FILES=$(find "$TARGET_PATH" \
    -type f \
    \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cuh" \) \
    ! -path "*/bazel-*/*" \
    ! -path "*/build/*" \
    ! -path "*/third_party/*" \
    ! -path "*/tdLBGeometryRushtonTurbineLib/*")

if [ -z "$CPP_FILES" ]; then
    echo -e "${YELLOW}No C++ files found${NC}"
    exit 0
fi

FILE_COUNT=$(echo "$CPP_FILES" | wc -l | tr -d ' ')
echo "Found $FILE_COUNT C++ files to process"
echo ""

if [ "$MODE" = "check" ]; then
    echo -e "${YELLOW}Running in CHECK mode (use --fix to apply changes)${NC}"
    echo ""

    FAILED=0
    for file in $CPP_FILES; do
        if ! clang-format --dry-run --Werror "$file" 2>&1 | grep -q "error:"; then
            # File is correctly formatted
            continue
        else
            echo -e "${RED}✗${NC} $file needs formatting"
            FAILED=$((FAILED + 1))
        fi
    done

    echo ""
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ All files are correctly formatted!${NC}"
        exit 0
    else
        echo -e "${RED}✗ $FAILED file(s) need formatting${NC}"
        echo "Run './format-code.sh --fix' to fix them"
        exit 1
    fi
else
    echo -e "${GREEN}Running in FIX mode${NC}"
    echo ""

    FORMATTED=0
    for file in $CPP_FILES; do
        # Check if file needs formatting
        if clang-format --dry-run --Werror "$file" 2>&1 | grep -q "error:"; then
            echo -e "${YELLOW}Formatting${NC} $file"
            clang-format -i "$file"
            FORMATTED=$((FORMATTED + 1))
        fi
    done

    echo ""
    if [ $FORMATTED -eq 0 ]; then
        echo -e "${GREEN}✓ No files needed formatting${NC}"
    else
        echo -e "${GREEN}✓ Formatted $FORMATTED file(s)${NC}"
    fi
fi
