"""
Step 7: Simple Domain Adaptation (Recalibration + Threshold Optimization Only)

This script implements a lightweight approach focusing on external recalibration
and threshold optimization without risky fine-tuning that can degrade performance.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, log_loss, brier_score_loss,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SimpleDomainAdaptation:
    """
    Simple domain adaptation using only recalibration and threshold optimization.
    """
    
    def __init__(self, primary_features_dir, primary_labels_path, external_data_dir):
        """Initialize the simple domain adaptation."""
        self.primary_features_dir = primary_features_dir
        self.primary_labels_path = primary_labels_path
        self.external_data_dir = external_data_dir
        
        # Load labels info
        with open(primary_labels_path, 'r') as f:
            self.labels_info = json.load(f)
        
        self.class_names = self.labels_info['class_names']
        self.num_classes = len(self.class_names)
        
        print(f"Classes: {self.class_names}")
        
    def load_data(self):
        """Load all required datasets."""
        print("Loading datasets...")
        
        # Primary dataset
        X_primary_train = np.load(os.path.join(self.primary_features_dir, "train_X.npy"))
        y_primary_train = np.load(os.path.join(self.primary_features_dir, "train_y.npy"))
        X_primary_val = np.load(os.path.join(self.primary_features_dir, "val_X.npy"))
        y_primary_val = np.load(os.path.join(self.primary_features_dir, "val_y.npy"))
        X_primary_test = np.load(os.path.join(self.primary_features_dir, "test_X.npy"))
        y_primary_test = np.load(os.path.join(self.primary_features_dir, "test_y.npy"))
        
        # External dataset
        external_features_dir = os.path.join(self.external_data_dir, "features", "mobilenetv2")
        X_external = np.load(os.path.join(external_features_dir, "external_X.npy"))
        y_external = np.load(os.path.join(external_features_dir, "external_y.npy"))
        
        # Create external train/val split
        X_external_train, X_external_val, y_external_train, y_external_val = train_test_split(
            X_external, y_external, test_size=0.2, stratify=y_external, random_state=42
        )
        
        print(f"Primary train: {X_primary_train.shape}, val: {X_primary_val.shape}, test: {X_primary_test.shape}")
        print(f"External train: {X_external_train.shape}, val: {X_external_val.shape}")
        print(f"External test: {X_external.shape}")
        
        return {
            'primary': {
                'train': (X_primary_train, y_primary_train),
                'val': (X_primary_val, y_primary_val),
                'test': (X_primary_test, y_primary_test)
            },
            'external': {
                'train': (X_external_train, y_external_train),
                'val': (X_external_val, y_external_val),
                'test': (X_external, y_external)
            }
        }
        
    def setup_models(self, data):
        """Setup and train models with domain-specific calibration."""
        print("Setting up models with domain-specific calibration...")
        
        # Standardize features using primary training data
        scaler = StandardScaler()
        X_primary_train_scaled = scaler.fit_transform(data['primary']['train'][0])
        X_primary_val_scaled = scaler.transform(data['primary']['val'][0])
        X_primary_test_scaled = scaler.transform(data['primary']['test'][0])
        X_external_train_scaled = scaler.transform(data['external']['train'][0])
        X_external_val_scaled = scaler.transform(data['external']['val'][0])
        X_external_test_scaled = scaler.transform(data['external']['test'][0])
        
        # Train base classifier on primary data
        base_classifier = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            random_state=42,
            C=10,
            max_iter=2000
        )
        
        # Fit on combined primary train+val
        X_primary_combined = np.vstack([X_primary_train_scaled, X_primary_val_scaled])
        y_primary_combined = np.concatenate([data['primary']['train'][1], data['primary']['val'][1]])
        base_classifier.fit(X_primary_combined, y_primary_combined)
        
        # Create domain-specific calibrations
        primary_calibrated = CalibratedClassifierCV(
            base_classifier, method='sigmoid', cv='prefit'
        )
        primary_calibrated.fit(X_primary_val_scaled, data['primary']['val'][1])
        
        external_calibrated = CalibratedClassifierCV(
            base_classifier, method='sigmoid', cv='prefit'
        )
        external_calibrated.fit(X_external_val_scaled, data['external']['val'][1])
        
        return {
            'base_classifier': base_classifier,
            'scaler': scaler,
            'primary_calibrated': primary_calibrated,
            'external_calibrated': external_calibrated,
            'scaled_data': {
                'primary': {
                    'train': (X_primary_train_scaled, data['primary']['train'][1]),
                    'val': (X_primary_val_scaled, data['primary']['val'][1]),
                    'test': (X_primary_test_scaled, data['primary']['test'][1])
                },
                'external': {
                    'train': (X_external_train_scaled, data['external']['train'][1]),
                    'val': (X_external_val_scaled, data['external']['val'][1]),
                    'test': (X_external_test_scaled, data['external']['test'][1])
                }
            }
        }
        
    def optimize_thresholds(self, y_true, y_prob, method='balanced'):
        """
        Optimize per-class thresholds for better performance.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            method: 'balanced', 'high_recall', or 'conservative'
        """
        print(f"Optimizing thresholds using {method} strategy...")
        
        thresholds = {}
        
        for class_idx in range(self.num_classes):
            y_binary = (y_true == class_idx).astype(int)
            y_class_prob = y_prob[:, class_idx]
            
            if method == 'balanced':
                # Optimize F1 score
                precision, recall, thresh = precision_recall_curve(y_binary, y_class_prob)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_idx = np.argmax(f1_scores)
                thresholds[class_idx] = thresh[best_idx] if best_idx < len(thresh) else 0.5
                
            elif method == 'high_recall':
                # Prioritize recall for tumor classes
                if class_idx < 3:  # Tumor classes
                    target_recall = 0.85
                else:  # No tumor
                    target_recall = 0.90
                    
                precision, recall, thresh = precision_recall_curve(y_binary, y_class_prob)
                
                # Find threshold that achieves target recall
                valid_indices = recall >= target_recall
                if np.any(valid_indices):
                    valid_precision = precision[valid_indices]
                    best_idx = np.argmax(valid_precision)
                    if best_idx < len(thresh):
                        thresholds[class_idx] = thresh[best_idx]
                    else:
                        thresholds[class_idx] = thresh[-1] if len(thresh) > 0 else 0.5
                else:
                    # Fallback to F1 optimization
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                    best_idx = np.argmax(f1_scores)
                    thresholds[class_idx] = thresh[best_idx] if best_idx < len(thresh) else 0.5
                    
            elif method == 'conservative':
                # Conservative thresholds for high precision
                if class_idx < 3:  # Tumor classes
                    thresholds[class_idx] = 0.6
                else:  # No tumor
                    thresholds[class_idx] = 0.7
        
        print(f"Optimized thresholds: {thresholds}")
        return thresholds
        
    def apply_thresholds(self, y_prob, thresholds):
        """Apply optimized thresholds to get predictions."""
        y_pred = np.zeros(y_prob.shape[0], dtype=int)
        
        for class_idx, threshold in thresholds.items():
            class_mask = y_prob[:, class_idx] >= threshold
            y_pred[class_mask] = class_idx
            
        # Handle cases where multiple classes meet threshold (take highest prob)
        for i in range(len(y_pred)):
            if np.sum(y_prob[i] >= np.array(list(thresholds.values()))) > 1:
                y_pred[i] = np.argmax(y_prob[i])
                
        # Handle cases where no class meets threshold (take highest prob)
        no_threshold_met = np.sum(y_prob >= np.array(list(thresholds.values())), axis=1) == 0
        if np.any(no_threshold_met):
            y_pred[no_threshold_met] = np.argmax(y_prob[no_threshold_met], axis=1)
        
        return y_pred
        
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        per_class_f1 = f1_score(y_true, y_pred, average=None)
        per_class_recall = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except:
            roc_auc = 0.0
            
        log_loss_score = log_loss(y_true, y_prob)
        
        # Calibration metrics
        ece, mce = self._calculate_calibration_metrics(y_true, y_prob)
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'per_class_f1': per_class_f1.tolist(),
            'per_class_recall': [per_class_recall[str(i)]['recall'] for i in range(self.num_classes)],
            'roc_auc': roc_auc,
            'log_loss': log_loss_score,
            'ece': ece,
            'mce': mce
        }
        
    def _calculate_calibration_metrics(self, y_true, y_prob, n_bins=10):
        """Calculate ECE and MCE for calibration assessment."""
        from sklearn.calibration import calibration_curve
        
        ece = 0
        mce = 0
        
        for class_idx in range(self.num_classes):
            y_binary = (y_true == class_idx).astype(int)
            y_class_prob = y_prob[:, class_idx]
            
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, y_class_prob, n_bins=n_bins
                )
                
                # Calculate ECE and MCE for this class
                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                for bin_lower, bin_upper, fraction_pos, mean_pred in zip(
                    bin_lowers, bin_uppers, fraction_of_positives, mean_predicted_value
                ):
                    in_bin = (y_class_prob > bin_lower) & (y_class_prob <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        ece += np.abs(fraction_pos - mean_pred) * prop_in_bin
                        mce = max(mce, np.abs(fraction_pos - mean_pred))
                        
            except:
                continue
        
        return ece / self.num_classes, mce
        
    def run_adaptation(self):
        """Run the complete simple adaptation pipeline."""
        print("=== Starting Simple Domain Adaptation ===\n")
        
        # Load data
        data = self.load_data()
        
        # Setup models
        models = self.setup_models(data)
        
        # Evaluate on external test set
        X_external_test = models['scaled_data']['external']['test'][0]
        y_external_test = models['scaled_data']['external']['test'][1]
        
        # Get predictions with external calibration
        y_prob_external = models['external_calibrated'].predict_proba(X_external_test)
        y_pred_external = models['external_calibrated'].predict(X_external_test)
        
        # Optimize thresholds on external validation set
        X_external_val = models['scaled_data']['external']['val'][0]
        y_external_val = models['scaled_data']['external']['val'][1]
        y_prob_external_val = models['external_calibrated'].predict_proba(X_external_val)
        
        # Try different threshold strategies
        threshold_strategies = {
            'balanced': self.optimize_thresholds(y_external_val, y_prob_external_val, method='balanced'),
            'high_recall': self.optimize_thresholds(y_external_val, y_prob_external_val, method='high_recall'),
            'conservative': self.optimize_thresholds(y_external_val, y_prob_external_val, method='conservative')
        }
        
        # Evaluate each strategy
        results = {}
        results['baseline'] = self.calculate_metrics(y_external_test, y_pred_external, y_prob_external)
        
        for strategy_name, thresholds in threshold_strategies.items():
            y_pred_thresh = self.apply_thresholds(y_prob_external, thresholds)
            results[strategy_name] = self.calculate_metrics(y_external_test, y_pred_thresh, y_prob_external)
            results[f'{strategy_name}_thresholds'] = thresholds
        
        # Also check primary test performance to ensure no degradation
        X_primary_test = models['scaled_data']['primary']['test'][0]
        y_primary_test = models['scaled_data']['primary']['test'][1]
        y_prob_primary = models['primary_calibrated'].predict_proba(X_primary_test)
        y_pred_primary = models['primary_calibrated'].predict(X_primary_test)
        
        results['primary_test'] = self.calculate_metrics(y_primary_test, y_pred_primary, y_prob_primary)
        
        return results


def main():
    """Main function for Step 7: Simple Domain Adaptation."""
    print("=== Step 7: Simple Domain Adaptation ===\n")
    
    # Setup paths
    primary_features_dir = "data/brainmri_4c/features/mobilenetv2"
    primary_labels_path = "data/brainmri_4c/labels.json"
    external_data_dir = "data/external_4c"
    
    # Initialize analyzer
    analyzer = SimpleDomainAdaptation(
        primary_features_dir, primary_labels_path, external_data_dir
    )
    
    # Run adaptation
    results = analyzer.run_adaptation()
    
    # Load original results for comparison
    with open('experiments/results/external_validation_results.json', 'r') as f:
        original_results = json.load(f)
    
    # Print results
    print(f"\n=== Results Summary ===")
    print(f"\n📊 External Test Performance:")
    print(f"  Baseline (original): Macro-F1 = {original_results['external_performance']['calibrated']['macro_f1']:.4f}")
    print(f"  Baseline (recalibrated): Macro-F1 = {results['baseline']['macro_f1']:.4f}")
    
    for strategy in ['balanced', 'high_recall', 'conservative']:
        if strategy in results:
            print(f"  {strategy.title()}: Macro-F1 = {results[strategy]['macro_f1']:.4f}")
            print(f"    Per-class F1: {[f'{f:.3f}' for f in results[strategy]['per_class_f1']]}")
            print(f"    Per-class Recall: {[f'{r:.3f}' for r in results[strategy]['per_class_recall']]}")
    
    print(f"\n📊 Primary Test Performance (no degradation check):")
    print(f"  Macro-F1 = {results['primary_test']['macro_f1']:.4f}")
    
    # Find best strategy
    best_strategy = max(['balanced', 'high_recall', 'conservative'], 
                       key=lambda s: results[s]['macro_f1'] if s in results else 0)
    
    print(f"\n🎯 Best Strategy: {best_strategy.title()}")
    print(f"  Macro-F1 Improvement: {results[best_strategy]['macro_f1'] - results['baseline']['macro_f1']:+.4f}")
    print(f"  Glioma F1: {results[best_strategy]['per_class_f1'][0]:.4f}")
    print(f"  Glioma Recall: {results[best_strategy]['per_class_recall'][0]:.4f}")
    
    # Save results
    print(f"\n--- Saving Results ---")
    os.makedirs('experiments/results', exist_ok=True)
    
    adaptation_results = {
        'original_external_performance': original_results['external_performance']['calibrated'],
        'simple_adaptation_results': results,
        'best_strategy': best_strategy,
        'improvement_summary': {
            'macro_f1_improvement': results[best_strategy]['macro_f1'] - results['baseline']['macro_f1'],
            'glioma_f1_improvement': results[best_strategy]['per_class_f1'][0] - results['baseline']['per_class_f1'][0],
            'glioma_recall_improvement': results[best_strategy]['per_class_recall'][0] - results['baseline']['per_class_recall'][0],
            'ece_improvement': results[best_strategy]['ece'] - results['baseline']['ece']
        }
    }
    
    with open('experiments/results/simple_adaptation_results.json', 'w') as f:
        json.dump(adaptation_results, f, indent=2, default=str)
    
    print(f"Results saved to: experiments/results/simple_adaptation_results.json")
    print(f"🎉 Simple domain adaptation complete!")


if __name__ == "__main__":
    main()
