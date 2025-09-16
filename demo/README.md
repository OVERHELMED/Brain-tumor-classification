# 🧠 Brain MRI Tumor Classification - Interactive Web Demo

A real-time web interface for brain MRI tumor classification using our trained MobileNetV2-based model.

## 🎯 Features

- **🖼️ Interactive Image Upload**: Drag-and-drop or click to upload MRI images
- **⚡ Real-time Classification**: Instant tumor classification with confidence scores
- **📊 Detailed Results**: Class probabilities, confidence visualization, and risk assessment
- **🎨 Modern UI**: Professional medical-grade interface
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **🔍 Model Information**: Detailed architecture and performance metrics

## 🚀 Quick Start

### Option 1: One-Command Setup (Recommended)

```bash
cd demo
python start_demo.py
```

This will:
- ✅ Check Python version compatibility
- 📦 Create virtual environment automatically
- 🔧 Install all dependencies
- 🚀 Start the web server
- 🌐 Open your browser automatically

### Option 2: Manual Setup

1. **Navigate to demo directory**:
   ```bash
   cd demo
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv demo_venv
   ```

3. **Activate virtual environment**:
   ```bash
   # Windows
   demo_venv\Scripts\activate
   
   # macOS/Linux
   source demo_venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the demo**:
   ```bash
   python app.py
   ```

6. **Open your browser** and go to: `http://localhost:5000`

## 🖼️ How to Use

1. **Upload Image**: 
   - Drag and drop an MRI image onto the upload area, or
   - Click "Choose File" to browse and select an image

2. **Analyze**: 
   - Click "Analyze Image" to get the prediction

3. **View Results**:
   - See the predicted tumor class
   - Review confidence scores and probabilities
   - Check risk level assessment

## 📊 Supported Image Formats

- **PNG** - Portable Network Graphics
- **JPG/JPEG** - Joint Photographic Experts Group
- **BMP** - Bitmap
- **TIFF** - Tagged Image File Format

**Recommended**: 224x224 pixels (images will be automatically resized)

## 🎯 Classification Classes

The model classifies brain MRI images into 4 categories:

| Class | Description | Icon |
|-------|-------------|------|
| **Glioma** | Aggressive brain tumor from glial cells | 🔴 |
| **Meningioma** | Tumor from brain covering (meninges) | 🟠 |
| **Pituitary** | Tumor in the pituitary gland | 🟣 |
| **No Tumor** | Normal brain tissue | 🟢 |

## 🔧 API Endpoints

The demo provides RESTful API endpoints:

### Health Check
```http
GET /api/health
```
Returns server and model status.

### Model Information
```http
GET /api/model-info
```
Returns model architecture and performance details.

### Single Image Prediction
```http
POST /api/predict
Content-Type: multipart/form-data

{
  "file": <image_file>
}
```

### Batch Prediction
```http
POST /api/batch-predict
Content-Type: multipart/form-data

{
  "files": [<image_file1>, <image_file2>, ...]
}
```

## 📈 Model Performance

| Metric | Internal Testing | External Testing |
|--------|------------------|------------------|
| **Accuracy** | 96.2% | 72.3% → 77.9%* |
| **Macro-F1** | 95.9% | 67.5% → 78.4%* |
| **Inference Time** | 21.9 ± 2.4 ms | Same |
| **Model Size** | 2.22M parameters | 8.52 MB |

*After domain adaptation

## 🔒 Security Features

- **File Validation**: Type and size checking
- **Input Sanitization**: Secure file handling
- **CORS Support**: Cross-origin request handling
- **Error Handling**: Comprehensive error management

## 🛠️ Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Framework**: TensorFlow/Keras
- **Model**: MobileNetV2 + Classical Classifiers
- **Image Processing**: PIL/Pillow

## 📁 Project Structure

```
demo/
├── app.py                 # Flask backend application
├── templates/
│   └── index.html        # Frontend HTML template
├── requirements.txt       # Python dependencies
├── start_demo.py         # Automated setup script
├── uploads/              # Uploaded images (auto-created)
└── README.md            # This file
```

## ⚠️ Medical Disclaimer

**Important**: This AI tool is designed for research and educational purposes only. The classification results should not be used as a substitute for professional medical diagnosis, advice, or treatment.

**Always consult with qualified healthcare professionals** for medical decisions. The accuracy of the AI model may vary and should be validated by medical experts.

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Kill process using port 5000
   lsof -ti:5000 | xargs kill -9  # macOS/Linux
   netstat -ano | findstr :5000   # Windows
   ```

2. **Module not found**:
   ```bash
   # Make sure virtual environment is activated
   pip install -r requirements.txt
   ```

3. **Model not loading**:
   - Check if model file exists in expected location
   - Demo will use mock model if trained model not found

### System Requirements

- **Python**: 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 1GB free space
- **OS**: Windows, macOS, or Linux

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the demo thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Brain MRI Tumor Classification research study. Please refer to the main project license.

## 🙏 Acknowledgments

- **Research Paper**: "Compute-Efficient, Calibrated, and Explainable Brain MRI Tumor Classification with External Testing"
- **Model Architecture**: MobileNetV2 from TensorFlow/Keras
- **UI Framework**: Modern CSS3 with responsive design
- **Icons**: Font Awesome

---

**For questions or support**, please refer to the main project repository or contact the research team.

🧠 **Happy Analyzing!** 🎉
