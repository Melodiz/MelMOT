#!/usr/bin/env python3
"""
Simple test runner for MelMOT.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run tests for MelMOT."""
    print("Running MelMOT tests...")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
        print("✓ pytest found")
    except ImportError:
        print("✗ pytest not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"])
        print("✓ pytest installed")
    
    # Run tests
    test_dir = Path("tests")
    if test_dir.exists():
        print(f"\nRunning tests from {test_dir}...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(test_dir), 
            "-v", 
            "--tb=short"
        ])
        
        if result.returncode == 0:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed!")
            sys.exit(1)
    else:
        print("No tests directory found.")
    
    # Run example
    print("\nRunning example...")
    print("-" * 30)
    
    example_script = Path("examples/simple_tracking.py")
    if example_script.exists():
        try:
            result = subprocess.run([
                sys.executable, str(example_script)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Example ran successfully!")
                print("\nOutput:")
                print(result.stdout)
            else:
                print("✗ Example failed!")
                print("Error:", result.stderr)
        except Exception as e:
            print(f"✗ Error running example: {e}")
    else:
        print("No example script found.")
    
    print("\nTest run complete!")


if __name__ == "__main__":
    main()
