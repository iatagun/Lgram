#!/usr/bin/env python3
"""
Build and publish script for Centering-Lgram package.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command and return result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if check and result.returncode != 0:
        sys.exit(1)
    
    return result

def clean():
    """Clean build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    dirs_to_clean = ['build', 'dist', 'centering_lgram.egg-info', 'lgram.egg-info']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"   Removed {dir_name}/")
    
    # Clean __pycache__
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs[:]:
            if dir_name == '__pycache__':
                shutil.rmtree(os.path.join(root, dir_name))
                dirs.remove(dir_name)
                print(f"   Removed {os.path.join(root, dir_name)}")

def check_requirements():
    """Check if build requirements are installed."""
    print("🔍 Checking build requirements...")
    
    required_packages = ['build', 'twine', 'wheel']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install build twine wheel")
        return False
    
    print("✅ All build requirements satisfied")
    return True

def run_tests():
    """Run tests before building."""
    print("🧪 Running tests...")
    result = run_command("python -m pytest tests/ -v", check=False)
    
    if result.returncode != 0:
        print("⚠️  Tests failed, but continuing with build...")
    else:
        print("✅ All tests passed")

def build():
    """Build the package."""
    print("🏗️  Building package...")
    
    # Build using build module (modern way)
    run_command("python -m build")
    
    print("✅ Package built successfully")
    
    # List built files
    if os.path.exists('dist'):
        print("\n📦 Built files:")
        for file in os.listdir('dist'):
            size = os.path.getsize(f'dist/{file}') / 1024
            print(f"   {file} ({size:.1f} KB)")

def check_package():
    """Check package with twine."""
    print("🔍 Checking package...")
    run_command("python -m twine check dist/*")
    print("✅ Package check passed")

def upload_test():
    """Upload to TestPyPI."""
    print("🚀 Uploading to TestPyPI...")
    run_command("python -m twine upload --repository testpypi dist/*")
    print("✅ Uploaded to TestPyPI")

def upload_prod():
    """Upload to PyPI."""
    print("🚀 Uploading to PyPI...")
    
    response = input("⚠️  Are you sure you want to upload to PyPI? (y/N): ")
    if response.lower() != 'y':
        print("❌ Upload cancelled")
        return
    
    run_command("python -m twine upload dist/*")
    print("✅ Uploaded to PyPI")

def main():
    """Main build script."""
    if len(sys.argv) < 2:
        print("""
Usage: python build_script.py <command>

Commands:
  clean        - Clean build artifacts
  build        - Build the package
  test         - Run tests  
  check        - Check package with twine
  upload-test  - Upload to TestPyPI
  upload       - Upload to PyPI
  full         - Clean, build, check, and upload to TestPyPI
        """)
        return

    command = sys.argv[1]
    
    if command == 'clean':
        clean()
    
    elif command == 'build':
        if not check_requirements():
            return
        clean()
        build()
    
    elif command == 'test':
        run_tests()
    
    elif command == 'check':
        check_package()
    
    elif command == 'upload-test':
        upload_test()
    
    elif command == 'upload':
        upload_prod()
    
    elif command == 'full':
        if not check_requirements():
            return
        clean()
        run_tests()
        build()
        check_package()
        upload_test()
        print("\n🎉 Full build and test upload completed!")
        print("   Check your package at: https://test.pypi.org/project/lgram/")
    
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == '__main__':
    main()
