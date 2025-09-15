"""
Complete Step 3: Final evaluation of best model (Logistic Regression) on test set.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


def main():
    """Complete Step 3 with final test evaluation."""
    print("=== Completing Step 3: Final Test Evaluation ===\n")
    
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
    
    # Create final model (Logistic Regression with best parameters)
    print("\n--- Creating Final Logistic Regression Model ---")
    final_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        C=10,  # Best parameter from validation
        max_iter=2000
    )
    
    # Combine train and val for final training
    X_train_val = np.vstack([X_train_scaled, X_val_scaled])
    y_train_val = np.concatenate([y_train, y_val])
    
    print(f"Combined train+val: {X_train_val.shape}, {y_train_val.shape}")
    
    # Fit on combined data
    print("Fitting final model on train+val...")
    final_model.fit(X_train_val, y_train_val)
    
    # Predict on test set
    print("Evaluating on test set...")
    test_pred = final_model.predict(X_test_scaled)
    test_proba = final_model.predict_proba(X_test_scaled)
    
    # Calculate comprehensive metrics
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1_macro = f1_score(y_test, test_pred, average='macro')
    test_f1_micro = f1_score(y_test, test_pred, average='micro')
    
    print(f"\n--- Final Test Results ---")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test F1-macro: {test_f1_macro:.4f} ({test_f1_macro*100:.2f}%)")
    print(f"Test F1-micro: {test_f1_micro:.4f} ({test_f1_micro*100:.2f}%)")
    
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
        print(f"\nTest ROC-AUC (macro, one-vs-rest): {roc_auc_ovr:.4f} ({roc_auc_ovr*100:.2f}%)")
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {e}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, test_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Number of Samples'})
    plt.title('Confusion Matrix - Final Test Results\nLogistic Regression on MobileNetV2 Features', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/final_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: figs/final_confusion_matrix.png")
    
    # Save results
    os.makedirs('experiments/results', exist_ok=True)
    
    # Model comparison results
    comparison_data = [
        {
            'model': 'Logistic Regression',
            'val_f1_macro': 0.9965,
            'val_accuracy': 0.9965,
            'test_accuracy': test_accuracy,
            'test_f1_macro': test_f1_macro,
            'test_f1_micro': test_f1_micro,
            'best_params': {'C': 10, 'max_iter': 2000}
        },
        {
            'model': 'Linear SVM',
            'val_f1_macro': 0.9901,
            'val_accuracy': 0.9904,
            'best_params': {'C': 0.01}
        },
        {
            'model': 'RBF SVM',
            'val_f1_macro': 0.9909,
            'val_accuracy': 0.9913,
            'best_params': {'C': 10, 'gamma': 'scale'}
        }
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('experiments/results/model_comparison.csv', index=False)
    print(f"Model comparison saved to: experiments/results/model_comparison.csv")
    
    # Final results
    final_results = {
        'best_model': 'Logistic Regression',
        'model_parameters': {'C': 10, 'max_iter': 2000, 'multi_class': 'multinomial', 'solver': 'lbfgs'},
        'validation_metrics': {
            'accuracy': 0.9965,
            'f1_macro': 0.9965
        },
        'test_metrics': {
            'accuracy': float(test_accuracy),
            'f1_macro': float(test_f1_macro),
            'f1_micro': float(test_f1_micro),
            'roc_auc_ovr': float(roc_auc_ovr) if 'roc_auc_ovr' in locals() else None
        },
        'per_class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1_per_class.tolist(),
            'class_names': class_names
        },
        'dataset_info': {
            'total_samples': len(y_train_val) + len(y_test),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'num_classes': len(class_names),
            'feature_dimension': X_train.shape[1]
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open('experiments/results/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Final results saved to: experiments/results/final_results.json")
    
    # Summary
    print(f"\n=== Step 3 Complete! ===")
    print(f"✅ Best Model: Logistic Regression")
    print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"✅ Test F1-macro: {test_f1_macro*100:.2f}%")
    print(f"✅ Results saved to experiments/results/")
    print(f"✅ Confusion matrix saved to figs/")
    print(f"\n🎉 Ready for Step 4: Model calibration and analysis!")


if __name__ == "__main__":
    main()
