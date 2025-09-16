#!/usr/bin/env python3
"""
Test script for Brain MRI Tumor Classification Demo
Validates API endpoints and functionality
"""

import requests
import json
import time
import os
from pathlib import Path

BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("❌ Cannot connect to demo server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("🔍 Testing model info...")
    try:
        response = requests.get(f"{BASE_URL}/api/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Architecture: {data.get('architecture')}")
            print(f"   Classes: {data.get('classes')}")
            print(f"   Parameters: {data.get('parameters')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def create_test_image():
    """Create a simple test image for prediction"""
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple test image (224x224 RGB)
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(test_image)
        
        test_path = "test_mri.png"
        image.save(test_path)
        print(f"✅ Created test image: {test_path}")
        return test_path
    except Exception as e:
        print(f"❌ Error creating test image: {e}")
        return None

def test_prediction(image_path):
    """Test the prediction endpoint"""
    print("🔍 Testing prediction...")
    try:
        if not os.path.exists(image_path):
            print(f"❌ Test image not found: {image_path}")
            return False
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                prediction = data.get('prediction', {})
                print(f"✅ Prediction successful")
                print(f"   Class: {prediction.get('class')}")
                print(f"   Confidence: {prediction.get('confidence', 0):.3f}")
                print(f"   Inference time: {data.get('inference_time_ms')}ms")
                return True
            else:
                print(f"❌ Prediction failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Prediction request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_invalid_file():
    """Test prediction with invalid file"""
    print("🔍 Testing invalid file handling...")
    try:
        # Create a text file instead of image
        with open("test_invalid.txt", "w") as f:
            f.write("This is not an image")
        
        with open("test_invalid.txt", 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/api/predict", files=files)
        
        if response.status_code == 400:
            print("✅ Invalid file properly rejected")
            return True
        else:
            print(f"❌ Invalid file not properly handled: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Invalid file test error: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists("test_invalid.txt"):
            os.remove("test_invalid.txt")

def cleanup_test_files():
    """Clean up test files"""
    test_files = ["test_mri.png", "test_invalid.txt"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🧹 Cleaned up: {file}")

def main():
    """Run all tests"""
    print("🧠 Brain MRI Tumor Classification Demo - Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Health check
    if test_health_check():
        tests_passed += 1
    print()
    
    # Test 2: Model info
    if test_model_info():
        tests_passed += 1
    print()
    
    # Test 3: Create test image
    test_image_path = create_test_image()
    if test_image_path:
        tests_passed += 1
    print()
    
    # Test 4: Prediction
    if test_image_path and test_prediction(test_image_path):
        tests_passed += 1
    print()
    
    # Test 5: Invalid file handling
    if test_invalid_file():
        tests_passed += 1
    print()
    
    # Results
    print("=" * 60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Demo is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    # Cleanup
    cleanup_test_files()
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
