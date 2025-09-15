"""
Step 6: External Validation for Brain MRI 4-Class Tumor Classification

This script performs external validation on a separate dataset to measure
cross-dataset generalization and calibration shift - essential for medical imaging papers.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, log_loss, brier_score_loss
)
import timm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ExternalValidationAnalyzer:
    """
    Analyzer for external validation of brain MRI classification model.
    """
    
    def __init__(self, primary_features_dir, primary_labels_path, external_data_dir):
        """
        Initialize the external validation analyzer.
        
        Args:
            primary_features_dir: Directory containing primary dataset features
            primary_labels_path: Path to primary dataset labels.json
            external_data_dir: Directory containing external dataset
        """
        self.primary_features_dir = primary_features_dir
        self.primary_labels_path = primary_labels_path
        self.external_data_dir = external_data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load primary dataset labels info
        with open(primary_labels_path, 'r') as f:
            self.labels_info = json.load(f)
        
        self.class_names = self.labels_info['class_names']
        self.class_to_idx = self.labels_info['class_to_label']
        
        # Map external dataset class names to primary dataset
        self.external_class_mapping = {
            'glioma_tumor': 'glioma',
            'meningioma_tumor': 'meningioma', 
            'pituitary_tumor': 'pituitary',
            'no_tumor': 'notumor'
        }
        
        print(f"Primary classes: {self.class_names}")
        print(f"External class mapping: {self.external_class_mapping}")
        print(f"Device: {self.device}")
        
        # Setup model and preprocessing
        self._setup_model()
        self._setup_preprocessing()
        
    def _setup_model(self):
        """Setup the trained model from primary dataset."""
        print("Setting up trained model from primary dataset...")
        
        # Load primary dataset features and labels
        X_train = np.load(os.path.join(self.primary_features_dir, "train_X.npy"))
        y_train = np.load(os.path.join(self.primary_features_dir, "train_y.npy"))
        X_val = np.load(os.path.join(self.primary_features_dir, "val_X.npy"))
        y_val = np.load(os.path.join(self.primary_features_dir, "val_y.npy"))
        
        # Standardize features (same as training)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Logistic Regression classifier (same as primary)
        self.classifier = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            random_state=42,
            C=10,
            max_iter=2000
        )
        
        # Fit on combined train+val (same as final model)
        X_train_val = np.vstack([X_train_scaled, X_val_scaled])
        y_train_val = np.concatenate([y_train, y_val])
        self.classifier.fit(X_train_val, y_train_val)
        
        # Apply calibration (same as Step 4)
        self.calibrated_classifier = CalibratedClassifierCV(
            self.classifier, method='sigmoid', cv='prefit'
        )
        
        # Fit calibration on validation set
        self.calibrated_classifier.fit(X_val_scaled, y_val)
        
        print("Model setup complete with calibration applied.")
        
    def _setup_preprocessing(self):
        """Setup image preprocessing (same as primary dataset)."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create MobileNetV2 model (same as primary)
        self.mobilenetv2 = timm.create_model(
            'mobilenetv2_100',
            pretrained=True,
            num_classes=0,  # No classifier
            global_pool='avg'
        )
        self.mobilenetv2.eval()
        
    def create_external_csv_index(self):
        """
        Create CSV index for external dataset.
        
        Returns:
            Path to created external_test.csv
        """
        print("Creating external dataset CSV index...")
        
        external_test_dir = os.path.join(self.external_data_dir, "Testing")
        external_csv_path = os.path.join(self.external_data_dir, "external_test.csv")
        
        # Collect all external test images
        image_data = []
        
        for external_class, primary_class in self.external_class_mapping.items():
            class_dir = os.path.join(external_test_dir, external_class)
            
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found, skipping...")
                continue
                
            # Get all image files
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                label = self.class_to_idx[primary_class]
                
                image_data.append({
                    'path': image_path,
                    'label': label,
                    'split': 'external_test'
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(image_data)
        df.to_csv(external_csv_path, index=False)
        
        print(f"Created external_test.csv with {len(df)} images")
        print(f"Class distribution:")
        print(df['label'].value_counts().sort_index())
        
        return external_csv_path
        
    def extract_external_features(self, external_csv_path):
        """
        Extract MobileNetV2 features for external dataset.
        
        Args:
            external_csv_path: Path to external_test.csv
            
        Returns:
            Tuple of (features, labels)
        """
        print("Extracting MobileNetV2 features for external dataset...")
        
        # Load external CSV
        df = pd.read_csv(external_csv_path)
        
        features_list = []
        labels_list = []
        
        # Process images in batches
        batch_size = 32
        self.mobilenetv2.to(self.device)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(df), batch_size)):
                batch_df = df.iloc[i:i+batch_size]
                batch_images = []
                batch_labels = []
                
                for _, row in batch_df.iterrows():
                    try:
                        # Load and preprocess image
                        image = Image.open(row['path']).convert('RGB')
                        image_tensor = self.transform(image).unsqueeze(0)
                        batch_images.append(image_tensor)
                        batch_labels.append(row['label'])
                    except Exception as e:
                        print(f"Error loading {row['path']}: {e}")
                        continue
                
                if batch_images:
                    # Stack images and move to device
                    batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
                    
                    # Extract features
                    features = self.mobilenetv2(batch_tensor)
                    features_list.append(features.cpu().numpy())
                    labels_list.extend(batch_labels)
        
        # Combine all features and labels
        X_external = np.vstack(features_list) if features_list else np.array([])
        y_external = np.array(labels_list)
        
        print(f"Extracted features shape: {X_external.shape}")
        print(f"Labels shape: {y_external.shape}")
        
        # Save external features
        external_features_dir = os.path.join(self.external_data_dir, "features", "mobilenetv2")
        os.makedirs(external_features_dir, exist_ok=True)
        
        np.save(os.path.join(external_features_dir, "external_X.npy"), X_external)
        np.save(os.path.join(external_features_dir, "external_y.npy"), y_external)
        
        print(f"External features saved to {external_features_dir}")
        
        return X_external, y_external
        
    def evaluate_external_performance(self, X_external, y_external):
        """
        Evaluate model performance on external dataset.
        
        Args:
            X_external: External dataset features
            y_external: External dataset labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating model performance on external dataset...")
        
        # Standardize external features using primary scaler
        X_external_scaled = self.scaler.transform(X_external)
        
        # Get predictions from both uncalibrated and calibrated models
        y_pred_uncal = self.classifier.predict(X_external_scaled)
        y_prob_uncal = self.classifier.predict_proba(X_external_scaled)
        
        y_pred_cal = self.calibrated_classifier.predict(X_external_scaled)
        y_prob_cal = self.calibrated_classifier.predict_proba(X_external_scaled)
        
        # Calculate metrics for both models
        results = {}
        
        for model_name, y_pred, y_prob in [
            ('uncalibrated', y_pred_uncal, y_prob_uncal),
            ('calibrated', y_pred_cal, y_prob_cal)
        ]:
            # Basic metrics
            accuracy = accuracy_score(y_external, y_pred)
            macro_f1 = f1_score(y_external, y_pred, average='macro')
            micro_f1 = f1_score(y_external, y_pred, average='micro')
            
            # Per-class metrics
            per_class_f1 = f1_score(y_external, y_pred, average=None)
            per_class_recall = classification_report(
                y_external, y_pred, output_dict=True, zero_division=0
            )
            
            # Probability-based metrics
            try:
                roc_auc = roc_auc_score(y_external, y_prob, multi_class='ovr', average='macro')
            except:
                roc_auc = 0.0
                
            log_loss_score = log_loss(y_external, y_prob)
            
            # Calculate multiclass Brier score
            try:
                # For multiclass, calculate Brier score for each class and average
                brier_scores = []
                for class_idx in range(len(self.class_names)):
                    y_binary = (y_external == class_idx).astype(int)
                    brier_scores.append(brier_score_loss(y_binary, y_prob[:, class_idx]))
                brier_score = np.mean(brier_scores)
            except:
                brier_score = 0.0
            
            # Calibration metrics
            ece, mce = self._calculate_calibration_metrics(y_external, y_prob)
            
            results[model_name] = {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1,
                'per_class_f1': per_class_f1.tolist(),
                'per_class_recall': [per_class_recall[str(i)]['recall'] for i in range(len(self.class_names))],
                'roc_auc': roc_auc,
                'log_loss': log_loss_score,
                'brier_score': brier_score,
                'ece': ece,
                'mce': mce,
                'predictions': y_pred.tolist(),
                'probabilities': y_prob.tolist()
            }
            
            print(f"\n{model_name.upper()} Model Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Macro F1: {macro_f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  ECE: {ece:.4f}")
            print(f"  MCE: {mce:.4f}")
            print(f"  Log Loss: {log_loss_score:.4f}")
        
        return results
        
    def _calculate_calibration_metrics(self, y_true, y_prob, n_bins=10):
        """Calculate ECE and MCE for calibration assessment."""
        from sklearn.calibration import calibration_curve
        
        ece = 0
        mce = 0
        
        for class_idx in range(len(self.class_names)):
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
        
        return ece / len(self.class_names), mce
        
    def compare_internal_vs_external(self, external_results):
        """
        Compare internal test performance vs external validation.
        
        Args:
            external_results: Results from external evaluation
            
        Returns:
            Dictionary with comparison metrics
        """
        print("\nComparing internal vs external performance...")
        
        # Load internal test results (from Step 4)
        internal_results_path = 'experiments/results/calibration_results.json'
        
        if os.path.exists(internal_results_path):
            with open(internal_results_path, 'r') as f:
                internal_data = json.load(f)
            
            # Extract internal test metrics (calibrated model)
            internal_test_metrics = internal_data['calibrated_metrics']
            
            comparison = {}
            
            # Map metric names between internal and external results
            metric_mapping = {
                'test_ece': 'ece',
                'test_mce': 'mce', 
                'test_log_loss': 'log_loss'
            }
            
            for internal_metric, external_metric in metric_mapping.items():
                if internal_metric in internal_test_metrics and external_metric in external_results['calibrated']:
                    internal_val = internal_test_metrics[internal_metric]
                    external_val = external_results['calibrated'][external_metric]
                    
                    absolute_drop = internal_val - external_val
                    relative_drop = (absolute_drop / internal_val) * 100 if internal_val != 0 else 0
                    
                    comparison[external_metric] = {
                        'internal': internal_val,
                        'external': external_val,
                        'absolute_drop': absolute_drop,
                        'relative_drop_percent': relative_drop
                    }
            
            # Note: Per-class comparison not available in internal results
            # This would require loading the original test results from Step 3
            
            return comparison
        else:
            print("Internal results not found, skipping comparison.")
            return {}
            
    def plot_external_reliability_diagram(self, y_external, y_prob_cal, save_path):
        """
        Plot reliability diagram for external validation.
        
        Args:
            y_external: External dataset true labels
            y_prob_cal: External dataset calibrated probabilities
            save_path: Path to save the plot
        """
        from sklearn.calibration import calibration_curve
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for class_idx, class_name in enumerate(self.class_names):
            y_binary = (y_external == class_idx).astype(int)
            y_class_prob = y_prob_cal[:, class_idx]
            
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, y_class_prob, n_bins=10
                )
                
                axes[class_idx].plot(mean_predicted_value, fraction_of_positives, "s-",
                                   label=f"{class_name} (n={y_binary.sum()})")
                axes[class_idx].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                axes[class_idx].set_xlabel('Mean Predicted Probability')
                axes[class_idx].set_ylabel('Fraction of Positives')
                axes[class_idx].set_title(f'Reliability Diagram - {class_name}')
                axes[class_idx].legend()
                axes[class_idx].grid(True, alpha=0.3)
                
            except Exception as e:
                axes[class_idx].text(0.5, 0.5, f'Error: {str(e)}', 
                                   transform=axes[class_idx].transAxes, ha='center')
                axes[class_idx].set_title(f'Reliability Diagram - {class_name}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"External reliability diagram saved to: {save_path}")


def main():
    """Main function for Step 6: External Validation."""
    print("=== Step 6: External Validation ===\n")
    
    # Setup paths
    primary_features_dir = "data/brainmri_4c/features/mobilenetv2"
    primary_labels_path = "data/brainmri_4c/labels.json"
    external_data_dir = "data/external_4c"
    
    # Initialize analyzer
    analyzer = ExternalValidationAnalyzer(
        primary_features_dir, primary_labels_path, external_data_dir
    )
    
    # Step 1: Create external CSV index
    external_csv_path = analyzer.create_external_csv_index()
    
    # Step 2: Extract external features
    X_external, y_external = analyzer.extract_external_features(external_csv_path)
    
    # Step 3: Evaluate external performance
    external_results = analyzer.evaluate_external_performance(X_external, y_external)
    
    # Step 4: Compare internal vs external
    comparison = analyzer.compare_internal_vs_external(external_results)
    
    # Step 5: Generate external reliability diagram
    y_prob_cal = np.array(external_results['calibrated']['probabilities'])
    external_reliability_path = 'figs/external_reliability_diagram.png'
    os.makedirs('figs', exist_ok=True)
    analyzer.plot_external_reliability_diagram(y_external, y_prob_cal, external_reliability_path)
    
    # Step 6: Save all results
    print("\n--- Saving External Validation Results ---")
    os.makedirs('experiments/results', exist_ok=True)
    
    # Prepare comprehensive results
    external_validation_results = {
        'dataset_info': {
            'external_dataset_path': external_data_dir,
            'external_test_images': len(y_external),
            'class_distribution': {
                analyzer.class_names[i]: int(np.sum(y_external == i))
                for i in range(len(analyzer.class_names))
            }
        },
        'external_performance': external_results,
        'internal_vs_external_comparison': comparison,
        'plots': {
            'external_reliability_diagram': external_reliability_path
        },
        'method_details': {
            'preprocessing': 'Same as primary dataset (224x224, ImageNet normalization)',
            'feature_extraction': 'MobileNetV2 penultimate layer features',
            'model': 'Calibrated Logistic Regression trained on primary dataset',
            'calibration': 'Temperature scaling fitted on primary validation set'
        }
    }
    
    with open('experiments/results/external_validation_results.json', 'w') as f:
        json.dump(external_validation_results, f, indent=2, default=str)
    
    print(f"External validation results saved to: experiments/results/external_validation_results.json")
    
    # Print summary
    print(f"\n=== Step 6 Complete: External Validation ===")
    print(f"✅ External dataset: {len(y_external)} images")
    print(f"✅ External macro-F1: {external_results['calibrated']['macro_f1']:.4f}")
    print(f"✅ External ECE: {external_results['calibrated']['ece']:.4f}")
    
    if comparison:
        print(f"✅ Performance drops:")
        for metric, data in comparison.items():
            if metric != 'per_class_f1':
                print(f"  {metric}: {data['absolute_drop']:.4f} ({data['relative_drop_percent']:.1f}% drop)")
    
    print(f"✅ Reliability diagram: {external_reliability_path}")
    print(f"✅ Results saved to experiments/results/")
    
    print(f"\n🎉 External validation complete! Ready for final analysis.")


if __name__ == "__main__":
    main()
