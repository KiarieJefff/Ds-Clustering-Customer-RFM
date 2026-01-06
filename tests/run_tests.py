"""
Test Runner for Customer Segmentation Project

This script runs all tests and generates a coverage report.
"""

import unittest
import sys
import os
from io import StringIO

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run all test modules and return results."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create test runner with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    # Get output
    output = stream.getvalue()
    
    return result, output

def generate_test_report(result, output):
    """Generate a test report."""
    report = []
    report.append("TEST EXECUTION REPORT")
    report.append("=" * 50)
    report.append(f"Tests Run: {result.testsRun}")
    report.append(f"Failures: {len(result.failures)}")
    report.append(f"Errors: {len(result.errors)}")
    report.append(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    report.append("")
    
    if result.failures:
        report.append("FAILURES:")
        report.append("-" * 20)
        for test, traceback in result.failures:
            report.append(f"FAILED: {test}")
            report.append(traceback)
            report.append("")
    
    if result.errors:
        report.append("ERRORS:")
        report.append("-" * 20)
        for test, traceback in result.errors:
            report.append(f"ERROR: {test}")
            report.append(traceback)
            report.append("")
    
    if result.wasSuccessful():
        report.append("✅ ALL TESTS PASSED!")
    else:
        report.append("❌ SOME TESTS FAILED!")
    
    return "\n".join(report)

def main():
    """Main test runner function."""
    print("Running Customer Segmentation Test Suite")
    print("=" * 50)
    
    # Run tests
    result, output = run_all_tests()
    
    # Print output
    print(output)
    
    # Generate and save report
    report = generate_test_report(result, output)
    
    # Save report
    reports_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    report_path = os.path.join(reports_dir, 'test_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nTest report saved to: {report_path}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()
