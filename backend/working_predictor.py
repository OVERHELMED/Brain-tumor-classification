# Copy of the working predictor for backend integration
import os
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BrainMRIModel:
    """
    Brain MRI Tumor Classification Model
    Uses the exact research pipeline: timm MobileNetV2 + albumentations + scikit-learn
    """
    
    def __init__(self):
        self.feature_extractor = None
        self.classifier = None
        self.scaler = None
        self.transforms = None
        self.class_names = ['GLIOMA', 'MENINGIOMA', 'PITUITARY', 'NOTUMOR']
        self.class_descriptions = {
            'GLIOMA': 'Tumor arising from glial cells in brain tissue',
            'MENINGIOMA': 'Tumor arising from meninges (brain covering)',
            'PITUITARY': 'Tumor in the pituitary gland',
            'NOTUMOR': 'Normal brain tissue without tumor'
        }
        self.risk_levels = {
            'GLIOMA': 'HIGH',
            'MENINGIOMA': 'HIGH', 
            'PITUITARY': 'MEDIUM',
            'NOTUMOR': 'LOW'
        }
        
        # Load all components
        self._load_model()
    
    def _load_model(self):
        """Load all model components"""
        try:
            print("📦 Loading MobileNetV2 feature extractor (timm)...")
            
            # Load timm MobileNetV2 (matches research pipeline exactly)
            self.feature_extractor = timm.create_model(
                'mobilenetv2_100', 
                pretrained=True,
                num_classes=0  # Remove classifier head, keep features only
            )
            self.feature_extractor.eval()
            print("✅ MobileNetV2 feature extractor loaded")
            
            # Configure transforms (matches research pipeline exactly)
            self.transforms = A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
            print("✅ Image transforms configured")
            
            # Load trained classifier and scaler
            model_dir = Path(__file__).parent.parent / "experiments" / "models"
            
            classifier_path = model_dir / "logistic_regression_classifier.pkl"
            self.classifier = joblib.load(classifier_path)
            print(f"✅ Loaded classifier from: {classifier_path}")
            
            scaler_path = model_dir / "feature_scaler.pkl"
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Loaded scaler from: {scaler_path}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image using albumentations (matches research pipeline)"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            
            # Apply transforms
            transformed = self.transforms(image=image_np)
            tensor = transformed['image']
            
            # Add batch dimension
            return tensor.unsqueeze(0)
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            raise e
    
    def _extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract features using MobileNetV2"""
        try:
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
                return features.numpy().flatten()
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            raise e
    
    def predict_single(self, image_path: str) -> Dict[str, Any]:
        """
        Predict single image
        Returns formatted result dictionary
        """
        try:
            start_time = time.time()
            
            # Preprocess image
            image_tensor = self._preprocess_image(image_path)
            
            # Extract features
            features = self._extract_features(image_tensor)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction and probabilities
            prediction_idx = self.classifier.predict(features_scaled)[0]
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            
            # Format results
            predicted_class = self.class_names[prediction_idx]
            confidence = probabilities[prediction_idx] * 100
            
            inference_time = (time.time() - start_time) * 1000
            
            # Create probability dictionary
            prob_dict = {}
            for i, class_name in enumerate(self.class_names):
                prob_dict[class_name] = round(probabilities[i] * 100, 1)
            
            result = {
                'prediction': predicted_class,
                'confidence': round(confidence, 1),
                'risk_level': self.risk_levels[predicted_class],
                'description': self.class_descriptions[predicted_class],
                'probabilities': prob_dict,
                'inference_time_ms': round(inference_time, 2),
                'image_path': os.path.basename(image_path)
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise e

def main():
    """Test the model"""
    print("🧪 Testing Brain MRI Model...")
    print("=" * 50)
    
    try:
        # Initialize model
        model = BrainMRIModel()
        print("🎉 All components loaded successfully!")
        
        # Test with sample images if available
        test_dir = Path(__file__).parent.parent / "data" / "brainmri_4c" / "testing"
        if test_dir.exists():
            sample_images = list(test_dir.glob("*.jpg"))[:4]
            
            for img_path in sample_images:
                result = model.predict_single(str(img_path))
                print(f"\n🔍 {result['image_path']}")
                print(f"📊 Prediction: {result['prediction']} ({result['confidence']}%)")
                print(f"⏱️  Time: {result['inference_time_ms']}ms")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
