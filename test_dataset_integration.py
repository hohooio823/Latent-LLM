#!/usr/bin/env python3
"""
Test script to verify dataset downloading and integration functionality.
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dataset_downloader():
    """Test the DatasetDownloader class."""
    print("Testing DatasetDownloader...")
    
    try:
        from dataset_downloader import DatasetDownloader
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = DatasetDownloader(cache_dir=temp_dir, tokenizer_name="gpt2")
            
            # Test that directories are created
            assert os.path.exists(temp_dir)
            assert os.path.exists(os.path.join(temp_dir, "raw"))
            assert os.path.exists(os.path.join(temp_dir, "tok_gpt2"))
            
            print("✅ DatasetDownloader initialization works")
            
            # Test tokenizer loading
            tokenizer = downloader.load_tokenizer()
            assert tokenizer is not None
            assert tokenizer.name_or_path == "gpt2"
            
            print("✅ Tokenizer loading works")
            
            # Test dataset verification (should fail since no data is downloaded)
            result = downloader.verify_dataset(split="train")
            assert result == False
            
            print("✅ Dataset verification works (correctly returns False for missing data)")
            
    except Exception as e:
        print(f"❌ DatasetDownloader test failed: {e}")
        return False
    
    return True

def test_task_integration():
    """Test the Task class integration."""
    print("\nTesting Task integration...")
    
    try:
        from owt import Task
        import tempfile
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test ensure_dataset_available (should return False for missing data)
            result = Task.ensure_dataset_available(
                split="train", 
                cache_dir=temp_dir, 
                auto_download=False
            )
            assert result == False
            
            print("✅ Task.ensure_dataset_available works (correctly returns False for missing data)")
            
            # Test with auto_download=True (should fail gracefully since we don't have internet)
            result = Task.ensure_dataset_available(
                split="train", 
                cache_dir=temp_dir, 
                auto_download=True
            )
            # This should return False due to the mock or lack of internet
            print("✅ Task.ensure_dataset_available handles download errors gracefully")
            
    except Exception as e:
        print(f"❌ Task integration test failed: {e}")
        return False
    
    return True

def test_config_integration():
    """Test the configuration integration."""
    print("\nTesting configuration integration...")
    
    try:
        import config
        
        # Test that new config variables exist
        assert hasattr(config, 'DATASET_NAME')
        assert hasattr(config, 'DATASET_AUTO_DOWNLOAD')
        assert hasattr(config, 'DATASET_NUM_SAMPLES')
        assert hasattr(config, 'DATASET_TOKENIZER')
        assert hasattr(config, 'DATASET_NUM_WORKERS')
        
        # Test that config values are reasonable
        assert config.DATASET_NAME == "openwebtext"
        assert config.DATASET_AUTO_DOWNLOAD == True
        assert config.DATASET_TOKENIZER == "gpt2"
        assert config.DATASET_NUM_WORKERS == 4
        
        print("✅ Configuration integration works")
        
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")
        return False
    
    return True

def test_download_script():
    """Test the download_dataset.py script."""
    print("\nTesting download_dataset.py script...")
    
    try:
        import subprocess
        import tempfile
        
        # Test --list_datasets option
        result = subprocess.run([sys.executable, "download_dataset.py", "--list_datasets"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert "Available datasets:" in result.stdout
        assert "openwebtext" in result.stdout
        
        print("✅ download_dataset.py --list_datasets works")
        
        # Test help option
        result = subprocess.run([sys.executable, "download_dataset.py", "--help"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert "Download and preprocess datasets" in result.stdout
        
        print("✅ download_dataset.py --help works")
        
    except Exception as e:
        print(f"❌ download_dataset.py test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Running dataset integration tests...\n")
    
    tests = [
        test_dataset_downloader,
        test_task_integration,
        test_config_integration,
        test_download_script,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dataset integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)