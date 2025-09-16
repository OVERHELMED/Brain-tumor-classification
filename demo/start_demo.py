#!/usr/bin/env python3
"""
Startup script for Brain MRI Tumor Classification Demo
Automatically sets up environment and starts the web demo
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path("demo_venv")
    
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "demo_venv"], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False
    else:
        print("✅ Virtual environment already exists")
    
    return True

def get_pip_command():
    """Get the appropriate pip command for the current OS"""
    if os.name == 'nt':  # Windows
        return os.path.join("demo_venv", "Scripts", "pip")
    else:  # Unix/Linux/macOS
        return os.path.join("demo_venv", "bin", "pip")

def get_python_command():
    """Get the appropriate python command for the current OS"""
    if os.name == 'nt':  # Windows
        return os.path.join("demo_venv", "Scripts", "python")
    else:  # Unix/Linux/macOS
        return os.path.join("demo_venv", "bin", "python")

def install_dependencies():
    """Install required dependencies"""
    pip_cmd = get_pip_command()
    
    print("📦 Installing dependencies...")
    try:
        # Upgrade pip first
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_model_file():
    """Check if model file exists"""
    model_paths = [
        "../experiments/models/adapted_mobilenetv2.pth",
        "../experiments/models/best_model.h5",
        "model/mobilenetv2_model.h5"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Model file found: {path}")
            return True
    
    print("⚠️  No trained model found. Demo will use a mock model for demonstration.")
    print("   To use your actual trained model, place it in one of these locations:")
    for path in model_paths:
        print(f"   - {path}")
    return True

def start_flask_app():
    """Start the Flask application"""
    python_cmd = get_python_command()
    
    print("🚀 Starting Brain MRI Tumor Classification Demo...")
    print("📍 Demo will be available at: http://localhost:5000")
    print("🌐 Opening browser in 3 seconds...")
    
    try:
        # Start Flask app in background
        process = subprocess.Popen([python_cmd, "app.py"])
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open("http://localhost:5000")
        
        print("\n" + "="*50)
        print("🎉 DEMO IS RUNNING!")
        print("="*50)
        print("📱 Web Interface: http://localhost:5000")
        print("🔍 API Health Check: http://localhost:5000/api/health")
        print("📊 Model Info: http://localhost:5000/api/model-info")
        print("⚠️  Press Ctrl+C to stop the demo")
        print("="*50)
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
        try:
            process.terminate()
        except:
            pass
    except Exception as e:
        print(f"❌ Error starting demo: {e}")

def main():
    """Main startup function"""
    print("🧠 Brain MRI Tumor Classification Demo Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        return
    
    # Create virtual environment
    if not create_virtual_environment():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Check model file
    check_model_file()
    
    # Start the demo
    start_flask_app()

if __name__ == "__main__":
    main()
