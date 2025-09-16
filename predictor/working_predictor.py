#!/usr/bin/env python3
"""
Working Brain MRI Predictor - Uses Your Exact Research Pipeline
This uses the same feature extraction method that achieved 96% accuracy
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib
from sklearn.preprocessing import StandardScaler
import tempfile

class SingleImageDataset(Dataset):
    """Dataset for single image feature extraction"""
    
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_path)
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy for albumentations
        image_np = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image_np)
            image_tensor = transformed['image']
        else:
            # Default transform
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        
        return image_tensor, 0  # Dummy label

class WorkingBrainMRIPredictor:
    """
    Brain MRI predictor using your exact research methodology
    """
    
    def __init__(self, model_dir="../experiments/models"):
        self.model_dir = model_dir
        self.feature_extractor = None
        self.classifier = None
        self.scaler = None
        self.transform = None
        
        # Class information (matching your research exactly)
        self.class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']
        self.class_descriptions = {
            'glioma': 'Aggressive brain tumor originating from glial cells',
            'meningioma': 'Tumor arising from meninges (brain covering)',
            'pituitary': 'Tumor in the pituitary gland',
            'notumor': 'Normal brain tissue without tumor'
        }
        
        self.load_models()
    
    def load_models(self):
        """Load models using your exact research methodology"""
        print("🧠 Loading Brain MRI Model (Research Pipeline)...")
        
        try:
            # 1. Load feature extractor (using timm like in your research)
            print("📦 Loading MobileNetV2 feature extractor (timm)...")
            self.feature_extractor = timm.create_model(
                'mobilenetv2_100',
                pretrained=True,
                num_classes=0,  # Remove classifier head
                global_pool='avg'
            )
            self.feature_extractor.eval()
            print("✅ MobileNetV2 feature extractor loaded")
            
            # 2. Setup transforms (matching your research exactly)
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
            print("✅ Image transforms configured")
            
            # 3. Load trained classifier
            classifier_path = os.path.join(self.model_dir, "logistic_regression_classifier.pkl")
            if os.path.exists(classifier_path):
                self.classifier = joblib.load(classifier_path)
                print(f"✅ Loaded classifier from: {classifier_path}")
            else:
                print(f"❌ Classifier not found: {classifier_path}")
                return False
            
            # 4. Load scaler
            scaler_path = os.path.join(self.model_dir, "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Loaded scaler from: {scaler_path}")
            else:
                print(f"❌ Scaler not found: {scaler_path}")
                return False
            
            print("🎉 All components loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_features(self, image_path):
        """Extract features using your exact research method"""
        try:
            # Create dataset for single image
            dataset = SingleImageDataset(image_path, self.transform)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            # Extract features
            features_list = []
            
            with torch.no_grad():
                for batch_images, _ in dataloader:
                    # Extract features using timm model
                    features = self.feature_extractor(batch_images)
                    features_list.append(features.cpu().numpy())
            
            # Concatenate all features
            features = np.vstack(features_list)
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, image_path, show_details=True):
        """
        Predict tumor class using your exact research pipeline
        """
        if not all([self.feature_extractor, self.classifier, self.scaler]):
            print("❌ Model components not loaded!")
            return None
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
        
        try:
            from datetime import datetime
            start_time = datetime.now()
            
            # Step 1: Extract features (using your research method)
            features = self.extract_features(image_path)
            if features is None:
                return None
            
            # Step 2: Scale features (using your trained scaler)
            scaled_features = self.scaler.transform(features)
            
            # Step 3: Classify (using your trained classifier)
            probabilities = self.classifier.predict_proba(scaled_features)
            predicted_class_idx = np.argmax(probabilities[0])
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Prepare results
            predicted_class = self.class_names[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx]
            
            results = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': {
                    name: prob for name, prob in zip(self.class_names, probabilities[0])
                },
                'inference_time_ms': inference_time,
                'description': self.class_descriptions[predicted_class]
            }
            
            if show_details:
                self.display_results(results, image_path)
            
            return results
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def display_results(self, results, image_path):
        """Display prediction results"""
        print("\n" + "="*60)
        print("🧠 BRAIN MRI TUMOR CLASSIFICATION RESULTS")
        print("="*60)
        
        print(f"📁 Image: {os.path.basename(image_path)}")
        print(f"⏱️  Inference Time: {results['inference_time_ms']:.2f} ms")
        print()
        
        # Main prediction
        predicted_class = results['predicted_class'].upper()
        confidence = results['confidence']
        
        if predicted_class == 'NOTUMOR':
            icon = "🟢"
            risk = "LOW"
        elif confidence > 0.8:
            icon = "🔴"
            risk = "HIGH"
        else:
            icon = "🟡"
            risk = "MEDIUM"
        
        print(f"{icon} PREDICTION: {predicted_class}")
        print(f"📊 Confidence: {confidence:.1%}")
        print(f"⚠️  Risk Level: {risk}")
        print(f"📝 Description: {results['description']}")
        print()
        
        # Class probabilities
        print("📊 CLASS PROBABILITIES:")
        print("-" * 40)
        for class_name, prob in results['class_probabilities'].items():
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"{class_name.upper():12} {prob:.1%} |{bar}|")
        
        print("="*60)

def test_with_training_images():
    """Test with images from your training dataset"""
    print("🧪 Testing with Training Dataset Images")
    print("=" * 50)
    
    predictor = WorkingBrainMRIPredictor()
    
    # Test with specific images from each class
    test_cases = [
        ("../data/brainmri_4c/testing/glioma", "glioma"),
        ("../data/brainmri_4c/testing/meningioma", "meningioma"),
        ("../data/brainmri_4c/testing/pituitary", "pituitary"),
        ("../data/brainmri_4c/testing/notumor", "notumor")
    ]
    
    correct_predictions = 0
    total_predictions = 0
    
    for test_dir, true_class in test_cases:
        if os.path.exists(test_dir):
            # Get first image from this class
            images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if images:
                image_path = os.path.join(test_dir, images[0])
                print(f"\n📁 TRUE CLASS: {true_class.upper()}")
                
                result = predictor.predict(image_path, show_details=False)
                
                if result:
                    predicted = result['predicted_class']
                    confidence = result['confidence']
                    
                    if predicted == true_class:
                        print(f"✅ CORRECT: {predicted.upper()} ({confidence:.1%})")
                        correct_predictions += 1
                    else:
                        print(f"❌ WRONG: Predicted {predicted.upper()}, Expected {true_class.upper()}")
                        print(f"   Confidence: {confidence:.1%}")
                        print(f"   All probabilities:")
                        for name, prob in result['class_probabilities'].items():
                            print(f"     {name}: {prob:.3f}")
                    
                    total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n🎯 Test Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    if accuracy > 0.8:
        print("🎉 Excellent! Model is working correctly!")
    else:
        print("⚠️ Low accuracy - still debugging needed")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        image_path = sys.argv[1]
        predictor = WorkingBrainMRIPredictor()
        predictor.predict(image_path)
    else:
        # Test mode
        test_with_training_images()
        
        # Interactive mode
        print("\n" + "="*50)
        predictor = WorkingBrainMRIPredictor()
        
        while True:
            print("\nEnter MRI image path (or 'quit' to exit):")
            image_path = input("> ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                break
            
            if os.path.exists(image_path):
                predictor.predict(image_path)
            else:
                print("❌ File not found!")
