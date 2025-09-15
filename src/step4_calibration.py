"""
Step 4: Probability Calibration and Reliability Analysis

This script calibrates the Logistic Regression model probabilities and evaluates
calibration quality using ECE, MCE, and reliability diagrams for medical AI.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, 
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(y_true, y_prob, n_bins=15):
    """
    Calculate Maximum Calibration Error (MCE).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        MCE score (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce


def plot_reliability_diagram(y_true, y_prob, class_name, n_bins=15, save_path=None):
    """
    Plot reliability diagram for a single class.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities
        class_name: Name of the class
        n_bins: Number of bins
        save_path: Path to save the plot
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
             label=f"{class_name} (Reliability)", linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=2)
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'Reliability Diagram - {class_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Add ECE score to plot
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    plt.text(0.05, 0.95, f'ECE: {ece:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reliability diagram saved to: {save_path}")
    
    plt.show()


def plot_macro_reliability_diagram(y_true, y_prob_all, class_names, n_bins=15, save_path=None):
    """
    Plot macro-averaged reliability diagram across all classes.
    
    Args:
        y_true: True labels (multi-class)
        y_prob_all: Predicted probabilities for all classes
        class_names: List of class names
        n_bins: Number of bins
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    ece_scores = []
    
    # Plot reliability curve for each class
    for i, class_name in enumerate(class_names):
        # Convert to binary for this class
        y_binary = (y_true == i).astype(int)
        y_prob_class = y_prob_all[:, i]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, y_prob_class, n_bins=n_bins
        )
        
        ece = expected_calibration_error(y_binary, y_prob_class, n_bins)
        ece_scores.append(ece)
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                color=colors[i], label=f"{class_name} (ECE: {ece:.4f})", 
                linewidth=2, markersize=6)
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=2)
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Reliability Diagrams - All Classes', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Add macro-averaged ECE
    macro_ece = np.mean(ece_scores)
    plt.text(0.05, 0.95, f'Macro-averaged ECE: {macro_ece:.4f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Macro reliability diagram saved to: {save_path}")
    
    plt.show()
    
    return macro_ece, ece_scores


def analyze_threshold_performance(y_true, y_prob_all, class_names):
    """
    Analyze precision/recall at different probability thresholds.
    
    Args:
        y_true: True labels
        y_prob_all: Predicted probabilities for all classes
        class_names: List of class names
    
    Returns:
        Dictionary with threshold analysis results
    """
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    results = {}
    
    for threshold in thresholds:
        # Get predictions using threshold
        y_pred_thresh = np.argmax(y_prob_all, axis=1)
        
        # Only consider predictions where max probability >= threshold
        max_probs = np.max(y_prob_all, axis=1)
        confident_mask = max_probs >= threshold
        
        if confident_mask.sum() == 0:
            continue
            
        y_true_confident = y_true[confident_mask]
        y_pred_confident = y_pred_thresh[confident_mask]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_confident, y_pred_confident)
        f1_macro = f1_score(y_true_confident, y_pred_confident, average='macro')
        f1_weighted = f1_score(y_true_confident, y_pred_confident, average='weighted')
        
        # Per-class precision/recall
        precision_per_class = precision_score(y_true_confident, y_pred_confident, average=None)
        recall_per_class = recall_score(y_true_confident, y_pred_confident, average=None)
        
        results[threshold] = {
            'threshold': threshold,
            'n_confident': confident_mask.sum(),
            'coverage': confident_mask.sum() / len(y_true),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class
        }
    
    return results


def main():
    """Main function for Step 4: Probability Calibration."""
    print("=== Step 4: Probability Calibration and Reliability Analysis ===\n")
    
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
    
    # Standardize features (same as in Step 3)
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression model (same parameters as best model from Step 3)
    print("\n--- Training Logistic Regression Model ---")
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        C=10,
        max_iter=2000
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Get uncalibrated probabilities on validation and test sets
    print("Getting uncalibrated probabilities...")
    y_prob_val_uncalibrated = model.predict_proba(X_val_scaled)
    y_prob_test_uncalibrated = model.predict_proba(X_test_scaled)
    
    # Calculate uncalibrated metrics
    print("\n--- Uncalibrated Model Metrics ---")
    val_ece_uncalibrated = expected_calibration_error(
        (y_val == np.argmax(y_prob_val_uncalibrated, axis=1)).astype(int),
        np.max(y_prob_val_uncalibrated, axis=1)
    )
    val_mce_uncalibrated = maximum_calibration_error(
        (y_val == np.argmax(y_prob_val_uncalibrated, axis=1)).astype(int),
        np.max(y_prob_val_uncalibrated, axis=1)
    )
    
    test_ece_uncalibrated = expected_calibration_error(
        (y_test == np.argmax(y_prob_test_uncalibrated, axis=1)).astype(int),
        np.max(y_prob_test_uncalibrated, axis=1)
    )
    test_mce_uncalibrated = maximum_calibration_error(
        (y_test == np.argmax(y_prob_test_uncalibrated, axis=1)).astype(int),
        np.max(y_prob_test_uncalibrated, axis=1)
    )
    
    print(f"Validation ECE (uncalibrated): {val_ece_uncalibrated:.4f}")
    print(f"Validation MCE (uncalibrated): {val_mce_uncalibrated:.4f}")
    print(f"Test ECE (uncalibrated): {test_ece_uncalibrated:.4f}")
    print(f"Test MCE (uncalibrated): {test_mce_uncalibrated:.4f}")
    
    # Apply calibration using temperature scaling
    print("\n--- Applying Temperature Scaling Calibration ---")
    calibrated_model = CalibratedClassifierCV(
        model, method='sigmoid', cv='prefit'
    )
    
    # Fit calibration on validation set
    calibrated_model.fit(X_val_scaled, y_val)
    
    # Get calibrated probabilities on test set
    y_prob_test_calibrated = calibrated_model.predict_proba(X_test_scaled)
    
    # Calculate calibrated metrics
    print("\n--- Calibrated Model Metrics ---")
    test_ece_calibrated = expected_calibration_error(
        (y_test == np.argmax(y_prob_test_calibrated, axis=1)).astype(int),
        np.max(y_prob_test_calibrated, axis=1)
    )
    test_mce_calibrated = maximum_calibration_error(
        (y_test == np.argmax(y_prob_test_calibrated, axis=1)).astype(int),
        np.max(y_prob_test_calibrated, axis=1)
    )
    
    print(f"Test ECE (calibrated): {test_ece_calibrated:.4f}")
    print(f"Test MCE (calibrated): {test_mce_calibrated:.4f}")
    
    # Calculate Brier score and log loss
    test_brier_uncalibrated = brier_score_loss(
        (y_test == np.argmax(y_prob_test_uncalibrated, axis=1)).astype(int),
        np.max(y_prob_test_uncalibrated, axis=1)
    )
    test_brier_calibrated = brier_score_loss(
        (y_test == np.argmax(y_prob_test_calibrated, axis=1)).astype(int),
        np.max(y_prob_test_calibrated, axis=1)
    )
    
    test_logloss_uncalibrated = log_loss(y_test, y_prob_test_uncalibrated)
    test_logloss_calibrated = log_loss(y_test, y_prob_test_calibrated)
    
    print(f"Test Brier Score (uncalibrated): {test_brier_uncalibrated:.4f}")
    print(f"Test Brier Score (calibrated): {test_brier_calibrated:.4f}")
    print(f"Test Log Loss (uncalibrated): {test_logloss_uncalibrated:.4f}")
    print(f"Test Log Loss (calibrated): {test_logloss_calibrated:.4f}")
    
    # Create reliability diagrams
    print("\n--- Creating Reliability Diagrams ---")
    os.makedirs('figs/calibration', exist_ok=True)
    
    # Individual class reliability diagrams (uncalibrated)
    for i, class_name in enumerate(class_names):
        y_binary = (y_test == i).astype(int)
        y_prob_class = y_prob_test_uncalibrated[:, i]
        
        plot_reliability_diagram(
            y_binary, y_prob_class, f"{class_name} (Uncalibrated)",
            save_path=f'figs/calibration/reliability_{class_name.lower()}_uncalibrated.png'
        )
    
    # Individual class reliability diagrams (calibrated)
    for i, class_name in enumerate(class_names):
        y_binary = (y_test == i).astype(int)
        y_prob_class = y_prob_test_calibrated[:, i]
        
        plot_reliability_diagram(
            y_binary, y_prob_class, f"{class_name} (Calibrated)",
            save_path=f'figs/calibration/reliability_{class_name.lower()}_calibrated.png'
        )
    
    # Macro-averaged reliability diagrams
    macro_ece_uncalibrated, ece_scores_uncalibrated = plot_macro_reliability_diagram(
        y_test, y_prob_test_uncalibrated, class_names,
        save_path='figs/calibration/reliability_macro_uncalibrated.png'
    )
    
    macro_ece_calibrated, ece_scores_calibrated = plot_macro_reliability_diagram(
        y_test, y_prob_test_calibrated, class_names,
        save_path='figs/calibration/reliability_macro_calibrated.png'
    )
    
    # Threshold analysis
    print("\n--- Threshold Analysis ---")
    threshold_results = analyze_threshold_performance(y_test, y_prob_test_calibrated, class_names)
    
    print("Performance at different confidence thresholds:")
    for threshold, results in threshold_results.items():
        print(f"  Threshold {threshold}: Coverage={results['coverage']:.3f}, "
              f"Accuracy={results['accuracy']:.4f}, F1-macro={results['f1_macro']:.4f}")
    
    # Save all calibration results
    print("\n--- Saving Calibration Results ---")
    os.makedirs('experiments/results', exist_ok=True)
    
    calibration_results = {
        'uncalibrated_metrics': {
            'validation_ece': val_ece_uncalibrated,
            'validation_mce': val_mce_uncalibrated,
            'test_ece': test_ece_uncalibrated,
            'test_mce': test_mce_uncalibrated,
            'test_brier_score': test_brier_uncalibrated,
            'test_log_loss': test_logloss_uncalibrated,
            'macro_ece': macro_ece_uncalibrated,
            'class_ece_scores': ece_scores_uncalibrated
        },
        'calibrated_metrics': {
            'test_ece': test_ece_calibrated,
            'test_mce': test_mce_calibrated,
            'test_brier_score': test_brier_calibrated,
            'test_log_loss': test_logloss_calibrated,
            'macro_ece': macro_ece_calibrated,
            'class_ece_scores': ece_scores_calibrated
        },
        'improvement': {
            'ece_improvement': test_ece_uncalibrated - test_ece_calibrated,
            'mce_improvement': test_mce_uncalibrated - test_mce_calibrated,
            'brier_improvement': test_brier_uncalibrated - test_brier_calibrated,
            'logloss_improvement': test_logloss_uncalibrated - test_logloss_calibrated
        },
        'threshold_analysis': threshold_results,
        'class_names': class_names,
        'calibration_method': 'sigmoid_platt_scaling',
        'calibration_fit_set': 'validation',
        'evaluation_set': 'test'
    }
    
    with open('experiments/results/calibration_results.json', 'w') as f:
        json.dump(calibration_results, f, indent=2, default=str)
    
    print(f"Calibration results saved to: experiments/results/calibration_results.json")
    
    # Final summary
    print(f"\n=== Step 4 Complete: Calibration Analysis ===")
    print(f"✅ ECE Improvement: {test_ece_uncalibrated:.4f} → {test_ece_calibrated:.4f} "
          f"({test_ece_uncalibrated - test_ece_calibrated:.4f} improvement)")
    print(f"✅ MCE Improvement: {test_mce_uncalibrated:.4f} → {test_mce_calibrated:.4f} "
          f"({test_mce_uncalibrated - test_mce_calibrated:.4f} improvement)")
    print(f"✅ Reliability diagrams saved to figs/calibration/")
    print(f"✅ Threshold analysis completed")
    print(f"✅ Ready for Step 5: Explainability analysis!")


if __name__ == "__main__":
    main()
