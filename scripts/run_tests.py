#!/usr/bin/env python3
"""
Test runner script for the federated learning system.
Provides convenient commands to run different test categories.
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run federated learning system tests")
    parser.add_argument(
        "test_type",
        choices=[
            "unit", "integration", "security", "privacy", "performance", 
            "stress", "mock-sdr", "all", "fast", "coverage"
        ],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (where supported)"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    # Add common options
    if args.verbose:
        base_cmd.append("-v")
    
    if args.fail_fast:
        base_cmd.append("-x")
    
    if args.parallel:
        base_cmd.extend(["-n", "auto"])
    
    # Test-specific configurations
    test_configs = {
        "unit": {
            "markers": ["-m", "unit"],
            "description": "Unit Tests",
            "paths": ["tests/unit/"]
        },
        "integration": {
            "markers": ["-m", "integration"],
            "description": "Integration Tests",
            "paths": ["tests/integration/"]
        },
        "security": {
            "markers": ["-m", "security"],
            "description": "Security Tests",
            "paths": ["tests/security/"]
        },
        "privacy": {
            "markers": ["-m", "privacy"],
            "description": "Privacy Tests",
            "paths": ["tests/security/"]
        },
        "performance": {
            "markers": ["-m", "performance"],
            "description": "Performance Tests",
            "paths": ["tests/performance/"]
        },
        "stress": {
            "markers": ["-m", "stress"],
            "description": "Stress Tests",
            "paths": ["tests/performance/"]
        },
        "mock-sdr": {
            "markers": ["-m", "mock_sdr"],
            "description": "Mock SDR Hardware Tests",
            "paths": ["tests/unit/test_mock_sdr_hardware.py"]
        },
        "fast": {
            "markers": ["-m", "not slow and not performance and not stress"],
            "description": "Fast Tests (excluding slow, performance, and stress tests)",
            "paths": ["tests/"]
        },
        "all": {
            "markers": [],
            "description": "All Tests",
            "paths": ["tests/"]
        },
        "coverage": {
            "markers": ["-m", "not stress"],  # Exclude stress tests for coverage
            "description": "Coverage Tests",
            "paths": ["tests/"],
            "extra_args": ["--cov=src", "--cov-report=term-missing", "--cov-report=html"]
        }
    }
    
    config = test_configs[args.test_type]
    
    # Build command
    cmd = base_cmd.copy()
    
    # Add markers
    if config["markers"]:
        cmd.extend(config["markers"])
    
    # Add coverage options
    if args.coverage or args.test_type == "coverage":
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
        if args.html_report:
            cmd.append("--cov-report=html")
    
    # Add extra arguments if specified
    if "extra_args" in config:
        cmd.extend(config["extra_args"])
    
    # Add test paths
    cmd.extend(config["paths"])
    
    # Run the tests
    success = run_command(cmd, config["description"])
    
    if not success:
        sys.exit(1)
    
    # Additional reporting
    if args.test_type == "coverage" or args.coverage:
        print(f"\n📊 Coverage report generated:")
        print(f"   - Terminal: See output above")
        if args.html_report or args.test_type == "coverage":
            print(f"   - HTML: htmlcov/index.html")
    
    if args.test_type == "performance":
        print(f"\n⚡ Performance test results:")
        print(f"   - Check output above for timing and resource usage metrics")
        print(f"   - Consider running with --verbose for detailed metrics")
    
    if args.test_type == "stress":
        print(f"\n🔥 Stress test results:")
        print(f"   - Check output above for system behavior under extreme conditions")
        print(f"   - Monitor system resources during execution")
    
    print(f"\n🎉 {config['description']} completed successfully!")


if __name__ == "__main__":
    # Ensure we're in the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    main()