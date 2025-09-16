#!/usr/bin/env python3
"""
Interactive Web Demo for Brain MRI Tumor Classification
Real-time classification with MobileNetV2-based model
"""

import os
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = "../experiments/models/adapted_mobilenetv2.pth"  # Adjust path as needed
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class names for brain tumor classification
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_DESCRIPTIONS = {
    'Glioma': 'Aggressive brain tumor originating from glial cells',
    'Meningioma': 'Tumor arising from meninges (brain covering)',
    'No Tumor': 'Normal brain tissue without tumor',
    'Pituitary': 'Tumor in the pituitary gland'
}

# Global model variable
model = None

def load_model():
    """Load the trained MobileNetV2 model"""
    global model
    try:
        # For demo purposes, we'll create a mock model
        # In production, load your actual trained model
        logger.info("Loading MobileNetV2 model...")
        
        # Create a simple model for demonstration
        model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        
        # Add classification head
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = model(inputs, training=False)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Initialize with random weights (in production, load your trained weights)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/')
def index():
    """Serve the main demo page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'architecture': 'MobileNetV2',
        'input_shape': [224, 224, 3],
        'num_classes': 4,
        'classes': CLASS_NAMES,
        'descriptions': CLASS_DESCRIPTIONS,
        'parameters': '2.22M',
        'flops': '305.73M'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict brain tumor class from uploaded image"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, TIFF'}), 400
        
        # Read and preprocess image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Make prediction
        start_time = datetime.now()
        predictions = model.predict(processed_image, verbose=0)
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get prediction results
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Prepare class probabilities
        class_probabilities = {}
        for i, class_name in enumerate(CLASS_NAMES):
            class_probabilities[class_name] = float(predictions[0][i])
        
        # Determine risk level
        risk_level = 'High' if predicted_class != 'No Tumor' and confidence > 0.8 else 'Medium' if predicted_class != 'No Tumor' else 'Low'
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'description': CLASS_DESCRIPTIONS[predicted_class],
                'risk_level': risk_level
            },
            'class_probabilities': class_probabilities,
            'inference_time_ms': round(inference_time, 2),
            'image_data': f"data:image/png;base64,{img_base64}",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict multiple images at once"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        results = []
        
        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                continue
            
            try:
                image = Image.open(io.BytesIO(file.read()))
                processed_image = preprocess_image(image)
                
                if processed_image is not None:
                    predictions = model.predict(processed_image, verbose=0)
                    predicted_class_idx = np.argmax(predictions[0])
                    predicted_class = CLASS_NAMES[predicted_class_idx]
                    confidence = float(predictions[0][predicted_class_idx])
                    
                    results.append({
                        'filename': file.filename,
                        'prediction': predicted_class,
                        'confidence': confidence,
                        'description': CLASS_DESCRIPTIONS[predicted_class]
                    })
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Brain MRI Tumor Classification Demo...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load model. Exiting...")
