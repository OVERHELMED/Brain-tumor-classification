@echo off
echo 🚀 Starting Tony Stark Brain MRI API Server...
echo.
echo 🧠 Loading Neural Networks...
echo 🌐 API will be available at: http://127.0.0.1:8000
echo 📚 API Documentation at: http://127.0.0.1:8000/docs
echo ⚡ Press Ctrl+C to stop
echo.

cd /d "%~dp0"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
