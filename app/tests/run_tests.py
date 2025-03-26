#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import argparse
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def run_tests(test_path=None, verbose=False):
    """
    Run all tests in the specified path or current directory.
    
    Args:
        test_path (str): Path to the test directory or file
        verbose (bool): Display verbose output
    
    Returns:
        int: Number of failed tests
    """
    # Determine test directory
    if test_path:
        test_dir = Path(test_path)
    else:
        test_dir = Path(__file__).parent
    
    # Create a test loader
    loader = unittest.TestLoader()
    
    # Discover tests
    suite = loader.discover(str(test_dir), pattern="test_*.py")
    
    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    
    # Run the test suite
    result = runner.run(suite)
    
    # Return number of failures and errors
    return len(result.failures) + len(result.errors)

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run the trading bot test suite.")
    parser.add_argument(
        "--path", 
        type=str, 
        default=None,
        help="Path to the test directory or file"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Display verbose output"
    )
    
    args = parser.parse_args()
    
    # Run tests
    failures = run_tests(args.path, args.verbose)
    
    # Exit with appropriate code
    sys.exit(failures)

if __name__ == "__main__":
    main() 