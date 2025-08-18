#!/usr/bin/env python3
"""
Test runner for torch-operation-counter enhanced operation counters.
This script runs all test files to verify the enhanced operation counting is working correctly.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import torch_operation_counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from test_basic_operations import TestBasicOperations
from test_matrix_operations import TestMatrixOperations
from test_neural_network_operations import TestNeuralNetworkOperations
from test_indexing_operations import TestIndexingOperations


def run_all_tests():
    """Run all test suites"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBasicOperations,
        TestMatrixOperations,
        TestNeuralNetworkOperations,
        TestIndexingOperations,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Return success/failure
    return len(result.failures) + len(result.errors) == 0


if __name__ == "__main__":
    print("Running comprehensive tests for torch-operation-counter enhanced operation counters...")
    print("="*80)
    
    success = run_all_tests()
    
    if success:
        print("\nAll tests passed! The enhanced operation counters are working correctly.")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please check the output above for details.")
        sys.exit(1)
