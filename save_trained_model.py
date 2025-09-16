#!/usr/bin/env python3
"""
Save the trained classifier and scaler for the predictor
This recreates and saves the exact model from your research
"""

import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib

def save_trained_model():
    """Recreate and save the trained model components."""
    print("🔧 Recreating and Saving Trained Model...")
    
    # Paths
    features_dir = "data/brainmri_4c/features/mobilenetv2"
    labels_path = "data/brainmri_4c/labels.json"
    
    # Load labels info
    with open(labels_path, 'r') as f:
        labels_info = json.load(f)
    
    class_names = labels_info['class_names']
    print(f"Classes: {class_names}")
    
    # Load features and labels
    print("Loading features...")
    X_train = np.load(os.path.join(features_dir, "train_X.npy"))
    y_train = np.load(os.path.join(features_dir, "train_y.npy"))
    X_val = np.load(os.path.join(features_dir, "val_X.npy"))
    y_val = np.load(os.path.join(features_dir, "val_y.npy"))
    X_test = np.load(os.path.join(features_dir, "test_X.npy"))
    y_test = np.load(os.path.join(features_dir, "test_y.npy"))
    
    print(f"Features loaded:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val: {X_val.shape}, {y_val.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create final model (matching your research exactly)
    print("Creating final Logistic Regression model...")
    final_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        C=10,  # Best parameter from your research
        max_iter=2000
    )
    
    # Combine train and val for final training (matching your methodology)
    X_train_val = np.vstack([X_train_scaled, X_val_scaled])
    y_train_val = np.concatenate([y_train, y_val])
    
    print(f"Combined train+val: {X_train_val.shape}, {y_train_val.shape}")
    
    # Fit on combined data
    print("Fitting final model on train+val...")
    final_model.fit(X_train_val, y_train_val)
    
    # Test on holdout test set
    print("Evaluating on test set...")
    test_pred = final_model.predict(X_test_scaled)
    test_proba = final_model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1_macro = f1_score(y_test, test_pred, average='macro')
    
    print(f"\n--- Verification Results ---")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test F1-macro: {test_f1_macro:.4f} ({test_f1_macro*100:.2f}%)")
    
    # Save the trained model components
    os.makedirs("experiments/models", exist_ok=True)
    
    classifier_path = "experiments/models/logistic_regression_classifier.pkl"
    scaler_path = "experiments/models/feature_scaler.pkl"
    
    print(f"\nSaving model components...")
    joblib.dump(final_model, classifier_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"✅ Classifier saved to: {classifier_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # Verify saved models work
    print(f"\nVerifying saved models...")
    loaded_classifier = joblib.load(classifier_path)
    loaded_scaler = joblib.load(scaler_path)
    
    # Test prediction
    test_features = X_test_scaled[:5]  # First 5 test samples
    test_labels = y_test[:5]
    
    predictions = loaded_classifier.predict(test_features)
    probabilities = loaded_classifier.predict_proba(test_features)
    
    print(f"✅ Saved models work correctly!")
    print(f"Sample predictions: {predictions}")
    print(f"Sample probabilities shape: {probabilities.shape}")
    
    # Check class distribution in predictions
    unique, counts = np.unique(test_pred, return_counts=True)
    print(f"\nTest set prediction distribution:")
    for class_idx, count in zip(unique, counts):
        class_name = class_names[class_idx]
        percentage = count / len(test_pred) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    return True

if __name__ == "__main__":
    success = save_trained_model()
    if success:
        print(f"\n🎉 SUCCESS! Trained model saved correctly.")
        print(f"Now test with: cd predictor && python test_predictor.py")
    else:
        print(f"\n❌ Failed to save model.")
