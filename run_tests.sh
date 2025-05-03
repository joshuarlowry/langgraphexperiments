#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${YELLOW}$1${NC}\n"
}

# Run unit tests only
run_unit_tests() {
    print_header "Running unit tests..."
    python -m pytest tests/unit -v
}

# Run integration tests only
run_integration_tests() {
    print_header "Running integration tests..."
    python -m pytest tests/integration -v
}

# Run e2e tests only
run_e2e_tests() {
    print_header "Running end-to-end tests..."
    python -m pytest tests/e2e -v
}

# Run all tests with coverage
run_all_with_coverage() {
    print_header "Running all tests with coverage..."
    python -m pytest --cov=dnddice tests/ --cov-report=term-missing
}

# Show help
show_help() {
    echo "Usage: ./run_tests.sh [option]"
    echo "Options:"
    echo "  unit       - Run unit tests only"
    echo "  integration - Run integration tests only"
    echo "  e2e        - Run end-to-end tests only"
    echo "  all        - Run all tests"
    echo "  coverage   - Run all tests with coverage report"
    echo "  help       - Show this help message"
}

# Main logic
case "$1" in
    "unit")
        run_unit_tests
        ;;
    "integration")
        run_integration_tests
        ;;
    "e2e")
        run_e2e_tests
        ;;
    "all")
        print_header "Running all tests..."
        python -m pytest
        ;;
    "coverage")
        run_all_with_coverage
        ;;
    "help"|*)
        show_help
        ;;
esac 