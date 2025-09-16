# 🧠 Brain MRI Tumor Classification Predictor

A standalone predictor using your trained MobileNetV2 + Logistic Regression model for brain tumor classification.

## 🎯 What This Does

Uses your **actual trained model** (96.2% accuracy) to classify brain MRI images into:
- **Glioma** - Aggressive brain tumor from glial cells
- **Meningioma** - Tumor from brain covering (meninges)  
- **Pituitary** - Tumor in pituitary gland
- **No Tumor** - Normal brain tissue

## 🚀 Quick Start

### Single Image Prediction
```bash
# Interactive mode
python brain_mri_predictor.py

# Command line
python quick_predict.py path/to/your/mri_image.jpg
```

### Batch Prediction
```bash
python brain_mri_predictor.py path/to/image/directory --batch
```

### Test with Your Training Data
```bash
python test_predictor.py
```

## 📊 Model Details

### Architecture
- **Feature Extractor**: MobileNetV2 (PyTorch, adapted weights)
- **Classifier**: Logistic Regression (C=10, multinomial)
- **Preprocessing**: StandardScaler + ImageNet normalization

### Performance (From Your Research)
- **Internal Accuracy**: 96.2%
- **Macro-F1**: 95.9%
- **Inference Time**: ~22ms per image
- **Model Size**: 8.52 MB

### Files Used
- `adapted_mobilenetv2.pth` - Your trained PyTorch feature extractor
- `logistic_regression_classifier.pkl` - Your trained classifier  
- `feature_scaler.pkl` - Feature preprocessing scaler

## 💻 Usage Examples

### Example 1: Single Image
```python
from brain_mri_predictor import BrainMRIPredictor

predictor = BrainMRIPredictor()
result = predictor.predict("mri_image.jpg")

print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Example 2: Batch Processing
```python
predictor = BrainMRIPredictor()
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = predictor.predict_batch(image_paths)
```

## 🔧 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required:
- PyTorch 2.1.0
- torchvision 0.16.0
- scikit-learn 1.3.2
- Pillow, numpy, joblib

## 📁 Project Structure

```
predictor/
├── brain_mri_predictor.py    # Main predictor class
├── quick_predict.py          # Simple single-image prediction
├── test_predictor.py         # Test with training data
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## ⚠️ Medical Disclaimer

This AI tool is for research and educational purposes only. Results should not be used for medical diagnosis without consultation with qualified healthcare professionals.

## 🎯 Expected Performance

Based on your research results:
- **Training dataset images**: ~96% accuracy expected
- **External images**: ~72-78% accuracy (domain shift)
- **Google images**: Variable (depends on similarity to training data)

The model performs best on images similar to your training dataset format and quality.

---

**🧠 Ready to classify brain tumors with your trained model!** 🎉
