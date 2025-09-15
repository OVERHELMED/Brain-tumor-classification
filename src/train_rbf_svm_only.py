"""
Train only RBF SVM with optimized parameter grid for faster execution.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


def main():
    """Train RBF SVM with optimized parameter grid."""
    print("=== Training RBF SVM Only ===\n")
    
    # Setup paths
    features_dir = "data/brainmri_4c/features/mobilenetv2"
    labels_path = "data/brainmri_4c/labels.json"
    
    # Load labels info
    with open(labels_path, 'r') as f:
        labels_info = json.load(f)
    
    class_names = labels_info['class_names']
    print(f"Classes: {class_names}")
    
    # Load features
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
    
    # Optimized parameter grid for RBF SVM (smaller and more targeted)
    param_grid = {
        'C': [1, 10],  # Reduced from [0.1, 1, 10]
        'gamma': ['scale', 0.01]  # Reduced from ['scale', 0.01, 0.001]
    }
    
    print(f"Parameter grid: {param_grid}")
    print(f"Total combinations: {len(param_grid['C']) * len(param_grid['gamma'])}")
    
    # Create RBF SVM model
    base_model = SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    
    # Grid search with reduced CV folds for speed
    print("\nStarting grid search...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=2,  # Reduced from 3 to 2 folds
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_svm_rbf = grid_search.best_estimator_
    print(f"\nBest RBF SVM params: {grid_search.best_params_}")
    
    # Evaluate on validation set
    val_pred = best_svm_rbf.predict(X_val_scaled)
    val_f1_macro = f1_score(y_val, val_pred, average='macro')
    val_accuracy = accuracy_score(y_val, val_pred)
    
    print(f"Validation F1-macro: {val_f1_macro:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Compare with previous results
    print(f"\n--- Comparison with Previous Results ---")
    print(f"Logistic Regression: F1-macro = 0.9965")
    print(f"Linear SVM: F1-macro = 0.9901")
    print(f"RBF SVM: F1-macro = {val_f1_macro:.4f}")
    
    if val_f1_macro > 0.9965:
        print(f"🎉 RBF SVM is the BEST model!")
        best_model_name = "RBF SVM"
    else:
        print(f"Logistic Regression remains the best model.")
        best_model_name = "Logistic Regression"
    
    # Final evaluation on test set
    print(f"\n--- Final Test Evaluation ---")
    
    # Combine train and val for final training
    X_train_val = np.vstack([X_train_scaled, X_val_scaled])
    y_train_val = np.concatenate([y_train, y_val])
    
    print(f"Combined train+val: {X_train_val.shape}, {y_train_val.shape}")
    
    # Create final model with best parameters
    final_model = SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42,
        **grid_search.best_params_
    )
    
    # Fit on combined data
    final_model.fit(X_train_val, y_train_val)
    
    # Predict on test set
    test_pred = final_model.predict(X_test_scaled)
    test_proba = final_model.predict_proba(X_test_scaled)
    
    # Calculate comprehensive metrics
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1_macro = f1_score(y_test, test_pred, average='macro')
    test_f1_micro = f1_score(y_test, test_pred, average='micro')
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1-macro: {test_f1_macro:.4f}")
    print(f"Test F1-micro: {test_f1_micro:.4f}")
    
    # Per-class metrics
    print(f"\nPer-class metrics:")
    precision = precision_score(y_test, test_pred, average=None)
    recall = recall_score(y_test, test_pred, average=None)
    f1_per_class = f1_score(y_test, test_pred, average=None)
    
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1_per_class[i]:.4f}")
    
    # ROC-AUC
    try:
        roc_auc_ovr = roc_auc_score(y_test, test_proba, multi_class='ovr', average='macro')
        print(f"Test ROC-AUC (macro, one-vs-rest): {roc_auc_ovr:.4f}")
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {e}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - RBF SVM - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/confusion_matrix_rbf_svm.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    os.makedirs('experiments/results', exist_ok=True)
    
    results_data = {
        'model': 'RBF SVM',
        'best_params': grid_search.best_params_,
        'val_f1_macro': val_f1_macro,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'test_f1_macro': test_f1_macro,
        'test_f1_micro': test_f1_micro,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'class_names': class_names,
        'confusion_matrix': cm.tolist()
    }
    
    with open('experiments/results/rbf_svm_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n=== RBF SVM Training Complete ===")
    print(f"Best model: {best_model_name}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test F1-macro: {test_f1_macro:.4f}")
    print(f"Results saved to: experiments/results/rbf_svm_results.json")


if __name__ == "__main__":
    main()
