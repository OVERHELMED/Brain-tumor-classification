"""
Medical Image Validator - Stage 1 of Two-Stage Pipeline
Uses Hugging Face CLIP for zero-shot medical vs natural image classification
"""

from transformers import pipeline, AutoProcessor, AutoModel
from PIL import Image
import numpy as np
import cv2
from typing import Dict, Tuple, Any
import warnings
import torch

warnings.filterwarnings('ignore')

class MedicalImageValidator:
    """
    Hugging Face CLIP-based medical image validator
    Uses zero-shot classification to distinguish medical from natural images
    """
    
    def __init__(self):
        self.classifier = None
        self.loaded = False
        
        # Medical vs natural image labels for zero-shot classification
        self.candidate_labels = [
            "brain MRI scan medical imaging",
            "CT scan medical imaging", 
            "X-ray medical imaging",
            "medical scan radiological image",
            "natural photograph",
            "human face portrait",
            "landscape scenery",
            "everyday object"
        ]
        
        # Initialize the validator
        self._load_model()
    
    def _load_model(self):
        """Load Hugging Face CLIP model for zero-shot classification"""
        try:
            print("🔍 Loading Medical Image Validator (CLIP)...")
            
            # Load CLIP model for zero-shot image classification
            self.classifier = pipeline(
                "zero-shot-image-classification",
                model="openai/clip-vit-large-patch14"
            )
            
            print("✅ Medical Image Validator (CLIP) loaded successfully")
            self.loaded = True
            
        except Exception as e:
            print(f"❌ Failed to load Medical Image Validator: {e}")
            print("🔄 Falling back to heuristic-only validation...")
            self.loaded = False
    
    def _analyze_image_properties(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image properties for medical image characteristics"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Calculate image statistics (convert numpy types to Python types for JSON serialization)
        properties = {
            'mean_intensity': float(np.mean(gray)),
            'std_intensity': float(np.std(gray)),
            'intensity_range': float(np.max(gray) - np.min(gray)),
            'is_mostly_grayscale': self._is_grayscale_dominant(image),
            'has_medical_intensity_pattern': self._has_medical_pattern(gray),
            'aspect_ratio': float(image.width / image.height),
            'size': list(image.size)
        }
        
        return properties
    
    def _is_grayscale_dominant(self, image: Image.Image) -> bool:
        """Check if image is predominantly grayscale (medical images often are)"""
        if image.mode != 'RGB':
            return True
        
        # Convert to numpy
        img_array = np.array(image)
        
        # Calculate variance between RGB channels
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        # If channels are very similar, it's likely grayscale
        rgb_variance = float(np.var([np.mean(r), np.mean(g), np.mean(b)]))
        
        return rgb_variance < 50  # Much stricter threshold for grayscale detection
    
    def _has_medical_pattern(self, gray_image: np.ndarray) -> bool:
        """Check for typical medical image intensity patterns"""
        # Medical images often have:
        # 1. Lower overall intensity (darker backgrounds)
        # 2. Specific intensity distributions
        # 3. Less texture variation than natural images
        
        mean_intensity = float(np.mean(gray_image))
        intensity_std = float(np.std(gray_image))
        
        # Heuristics for medical-like patterns
        has_dark_background = mean_intensity < 100  # Darker overall
        has_moderate_contrast = 20 < intensity_std < 80  # Not too flat, not too busy
        
        return has_dark_background and has_moderate_contrast
    
    def validate_medical_image(self, image_path: str) -> Dict[str, Any]:
        """
        Validate if image is a medical image using CLIP zero-shot classification
        Returns confidence and reasoning
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Analyze basic image properties for additional context
            properties = self._analyze_image_properties(image)
            
            # CLIP-based validation (if model is loaded)
            if self.loaded and self.classifier:
                clip_results = self.classifier(image, candidate_labels=self.candidate_labels)
                
                # Calculate medical confidence from CLIP results
                medical_confidence = self._calculate_medical_confidence(clip_results)
                
                # Heuristic-based validation as backup
                heuristic_score = self._calculate_heuristic_score(properties)
                
                # Combine CLIP (primary) + heuristics (backup)
                final_confidence = float((medical_confidence * 0.8) + (heuristic_score * 0.2))
                is_medical = final_confidence > 0.6  # Medical threshold
                
                return {
                    'is_medical': bool(is_medical),
                    'confidence': final_confidence,
                    'medical_confidence': float(medical_confidence),
                    'heuristic_score': float(heuristic_score),
                    'clip_results': clip_results,
                    'properties': properties,
                    'reasoning': self._generate_clip_reasoning(clip_results, heuristic_score, final_confidence)
                }
            else:
                # Fallback to heuristic-only validation
                heuristic_score = self._calculate_heuristic_score(properties)
                is_medical = heuristic_score > 0.8  # Very strict for heuristic-only
                
                return {
                    'is_medical': bool(is_medical),
                    'confidence': float(heuristic_score),
                    'medical_confidence': 0.0,
                    'heuristic_score': float(heuristic_score),
                    'clip_results': [],
                    'properties': properties,
                    'reasoning': f"Heuristic-only validation (CLIP not loaded): {heuristic_score:.3f}"
                }
            
        except Exception as e:
            print(f"❌ Error validating image: {e}")
            return {
                'is_medical': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_heuristic_score(self, properties: Dict[str, Any]) -> float:
        """Calculate heuristic-based medical image score"""
        score = 0.0
        
        # Grayscale images are more likely to be medical
        if properties['is_mostly_grayscale']:
            score += 0.4
        
        # Medical intensity patterns (stricter check)
        if properties['has_medical_intensity_pattern']:
            score += 0.5
        
        # Reasonable aspect ratio (not too wide/tall)
        aspect_ratio = properties['aspect_ratio']
        if 0.8 <= aspect_ratio <= 1.2:  # More strict for square medical images
            score += 0.2
        
        # Intensity characteristics (stricter range)
        mean_intensity = properties['mean_intensity']
        if 20 <= mean_intensity <= 120:  # Tighter medical image range
            score += 0.1
        else:
            score -= 0.2  # Penalty for non-medical intensity ranges
        
        return float(max(0.0, min(score, 1.0)))  # Ensure score is between 0 and 1
    
    def _calculate_medical_confidence(self, clip_results: list) -> float:
        """Calculate medical confidence from CLIP zero-shot results"""
        try:
            # Sum scores for medical-related labels
            medical_score = 0.0
            natural_score = 0.0
            
            medical_keywords = ["MRI", "CT", "X-ray", "medical", "scan", "radiological"]
            natural_keywords = ["photograph", "face", "portrait", "landscape", "object"]
            
            for result in clip_results:
                label = result['label'].lower()
                score = result['score']
                
                # Check if label contains medical keywords
                if any(keyword.lower() in label for keyword in medical_keywords):
                    medical_score += score
                elif any(keyword.lower() in label for keyword in natural_keywords):
                    natural_score += score
            
            # Normalize to get medical confidence
            total_score = medical_score + natural_score
            if total_score > 0:
                medical_confidence = medical_score / total_score
            else:
                medical_confidence = 0.0
            
            return float(medical_confidence)
            
        except Exception as e:
            print(f"❌ CLIP confidence calculation error: {e}")
            return 0.0
    
    def _generate_clip_reasoning(self, clip_results: list, heuristic_score: float, final_confidence: float) -> str:
        """Generate human-readable reasoning from CLIP results"""
        try:
            # Get top prediction from CLIP
            top_prediction = clip_results[0] if clip_results else None
            
            reasons = []
            
            if top_prediction:
                top_label = top_prediction['label']
                top_score = top_prediction['score']
                
                if 'medical' in top_label.lower() or 'MRI' in top_label or 'scan' in top_label.lower():
                    reasons.append(f"CLIP identifies as '{top_label}' ({top_score:.1%})")
                else:
                    reasons.append(f"CLIP identifies as '{top_label}' ({top_score:.1%}) - non-medical")
            
            if heuristic_score > 0.6:
                reasons.append("Image properties align with medical imaging")
            elif heuristic_score < 0.3:
                reasons.append("Image properties suggest natural photograph")
            
            if final_confidence > 0.6:
                reasons.append("Strong medical image indicators")
            else:
                reasons.append("Insufficient medical image characteristics")
            
            return " | ".join(reasons) if reasons else "Validation completed"
            
        except Exception as e:
            return f"Reasoning generation error: {e}"

def main():
    """Test the medical image validator"""
    print("🧪 Testing Medical Image Validator")
    print("=" * 50)
    
    validator = MedicalImageValidator()
    
    # Test with a sample image (you can replace with actual path)
    test_image = "../data/brainmri_4c/testing/glioma/Te-gl_0000.jpg"
    
    if validator.loaded:
        result = validator.validate_medical_image(test_image)
        
        print(f"Image: {test_image}")
        print(f"Is Medical: {result['is_medical']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')}")
    else:
        print("❌ Validator not loaded properly")

if __name__ == "__main__":
    main()
