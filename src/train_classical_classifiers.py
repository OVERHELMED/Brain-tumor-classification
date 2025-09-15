"""
Train classical classifiers on MobileNetV2 features for Brain MRI dataset.

This script trains Logistic Regression and SVM classifiers on the extracted
MobileNetV2 embeddings, tunes hyperparameters, and evaluates performance.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')


class ClassicalClassifierTrainer:
    """
    Trainer for classical classifiers on extracted features.
    """
    
    def __init__(self, features_dir: str, labels_info: dict):
        """
        Initialize the trainer.
        
        Args:
            features_dir (str): Directory containing feature files
            labels_info (dict): Labels mapping information
        """
        self.features_dir = features_dir
        self.labels_info = labels_info
        self.class_names = labels_info['class_names']
        self.num_classes = labels_info['num_classes']
        
        # Load features
        self.X_train, self.y_train = self._load_features('train')
        self.X_val, self.y_val = self._load_features('val')
        self.X_test, self.y_test = self._load_features('test')
        
        print(f"Loaded features:")
        print(f"  Train: {self.X_train.shape}, {self.y_train.shape}")
        print(f"  Val: {self.X_val.shape}, {self.y_val.shape}")
        print(f"  Test: {self.X_test.shape}, {self.y_test.shape}")
        
        # Initialize scaler
        self.scaler = StandardScaler()
        self._fit_scaler()
        
        # Results storage
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def _load_features(self, split: str) -> tuple:
        """
        Load features and labels for a split.
        
        Args:
            split (str): Split name (train/val/test)
            
        Returns:
            tuple: (features, labels)
        """
        X_path = os.path.join(self.features_dir, f"{split}_X.npy")
        y_path = os.path.join(self.features_dir, f"{split}_y.npy")
        
        X = np.load(X_path)
        y = np.load(y_path)
        
        return X, y
    
    def _fit_scaler(self):
        """Fit StandardScaler on training data only."""
        print("Fitting StandardScaler on training data...")
        self.scaler.fit(self.X_train)
        
        # Transform all splits
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features standardized successfully")
    
    def train_logistic_regression(self):
        """Train and tune Logistic Regression."""
        print("\n--- Training Logistic Regression ---")
        
        # Define parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'max_iter': [2000]
        }
        
        # Create base model
        base_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            random_state=42
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Get best model
        best_lr = grid_search.best_estimator_
        print(f"Best Logistic Regression params: {grid_search.best_params_}")
        
        # Evaluate on validation set
        val_pred = best_lr.predict(self.X_val_scaled)
        val_f1_macro = f1_score(self.y_val, val_pred, average='macro')
        val_accuracy = accuracy_score(self.y_val, val_pred)
        
        print(f"Validation F1-macro: {val_f1_macro:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Store results
        self.results['logistic_regression'] = {
            'model': best_lr,
            'params': grid_search.best_params_,
            'val_f1_macro': val_f1_macro,
            'val_accuracy': val_accuracy
        }
        
        return best_lr, val_f1_macro
    
    def train_svm_linear(self):
        """Train and tune Linear SVM."""
        print("\n--- Training Linear SVM ---")
        
        # Define parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10]
        }
        
        # Create base model
        base_model = LinearSVC(
            class_weight='balanced',
            random_state=42,
            max_iter=2000
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Get best model
        best_svm_linear = grid_search.best_estimator_
        print(f"Best Linear SVM params: {grid_search.best_params_}")
        
        # Evaluate on validation set
        val_pred = best_svm_linear.predict(self.X_val_scaled)
        val_f1_macro = f1_score(self.y_val, val_pred, average='macro')
        val_accuracy = accuracy_score(self.y_val, val_pred)
        
        print(f"Validation F1-macro: {val_f1_macro:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Store results
        self.results['svm_linear'] = {
            'model': best_svm_linear,
            'params': grid_search.best_params_,
            'val_f1_macro': val_f1_macro,
            'val_accuracy': val_accuracy
        }
        
        return best_svm_linear, val_f1_macro
    
    def train_svm_rbf(self):
        """Train and tune RBF SVM."""
        print("\n--- Training RBF SVM ---")
        
        # Define parameter grid (smaller for speed)
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.001]
        }
        
        # Create base model
        base_model = SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Get best model
        best_svm_rbf = grid_search.best_estimator_
        print(f"Best RBF SVM params: {grid_search.best_params_}")
        
        # Evaluate on validation set
        val_pred = best_svm_rbf.predict(self.X_val_scaled)
        val_f1_macro = f1_score(self.y_val, val_pred, average='macro')
        val_accuracy = accuracy_score(self.y_val, val_pred)
        
        print(f"Validation F1-macro: {val_f1_macro:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Store results
        self.results['svm_rbf'] = {
            'model': best_svm_rbf,
            'params': grid_search.best_params_,
            'val_f1_macro': val_f1_macro,
            'val_accuracy': val_accuracy
        }
        
        return best_svm_rbf, val_f1_macro
    
    def select_best_model(self):
        """Select the best model based on validation F1-macro score."""
        print("\n--- Selecting Best Model ---")
        
        best_f1 = -1
        best_model_name = None
        
        for model_name, result in self.results.items():
            f1_score = result['val_f1_macro']
            print(f"{model_name}: F1-macro = {f1_score:.4f}")
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model_name = model_name
        
        self.best_model_name = best_model_name
        self.best_model = self.results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best validation F1-macro: {best_f1:.4f}")
        
        return best_model_name, best_f1
    
    def final_refit_and_evaluate(self):
        """Refit best model on train+val and evaluate on test."""
        print(f"\n--- Final Refit and Test Evaluation ---")
        
        # Combine train and val for final training
        X_train_val = np.vstack([self.X_train_scaled, self.X_val_scaled])
        y_train_val = np.concatenate([self.y_train, self.y_val])
        
        print(f"Combined train+val: {X_train_val.shape}, {y_train_val.shape}")
        
        # Create new instance of best model with same parameters
        if self.best_model_name == 'logistic_regression':
            final_model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                class_weight='balanced',
                random_state=42,
                **self.results[self.best_model_name]['params']
            )
        elif self.best_model_name == 'svm_linear':
            final_model = LinearSVC(
                class_weight='balanced',
                random_state=42,
                max_iter=2000,
                **self.results[self.best_model_name]['params']
            )
        elif self.best_model_name == 'svm_rbf':
            final_model = SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42,
                **self.results[self.best_model_name]['params']
            )
        
        # Fit on combined data
        final_model.fit(X_train_val, y_train_val)
        
        # Predict on test set
        test_pred = final_model.predict(self.X_test_scaled)
        
        # Get probability estimates if available
        test_proba = None
        if hasattr(final_model, 'predict_proba'):
            test_proba = final_model.predict_proba(self.X_test_scaled)
        elif hasattr(final_model, 'decision_function'):
            # For LinearSVC, we might need to calibrate
            try:
                test_proba = final_model.predict_proba(self.X_test_scaled)
            except AttributeError:
                print("Note: LinearSVC doesn't provide probability estimates")
        
        # Calculate comprehensive metrics
        self._evaluate_test_set(test_pred, test_proba)
        
        return final_model, test_pred, test_proba
    
    def _evaluate_test_set(self, test_pred, test_proba=None):
        """Evaluate model on test set with comprehensive metrics."""
        print(f"\n--- Test Set Evaluation ---")
        
        # Basic metrics
        test_accuracy = accuracy_score(self.y_test, test_pred)
        test_f1_macro = f1_score(self.y_test, test_pred, average='macro')
        test_f1_micro = f1_score(self.y_test, test_pred, average='micro')
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1-macro: {test_f1_macro:.4f}")
        print(f"Test F1-micro: {test_f1_micro:.4f}")
        
        # Per-class metrics
        print(f"\nPer-class metrics:")
        precision = precision_score(self.y_test, test_pred, average=None)
        recall = recall_score(self.y_test, test_pred, average=None)
        f1_per_class = f1_score(self.y_test, test_pred, average=None)
        
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1_per_class[i]:.4f}")
        
        # ROC-AUC if probabilities available
        if test_proba is not None:
            try:
                roc_auc_ovr = roc_auc_score(self.y_test, test_proba, multi_class='ovr', average='macro')
                print(f"Test ROC-AUC (macro, one-vs-rest): {roc_auc_ovr:.4f}")
            except Exception as e:
                print(f"Could not calculate ROC-AUC: {e}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, test_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, test_pred)
        self._plot_confusion_matrix(cm)
        
        # Store final results
        self.final_results = {
            'test_accuracy': test_accuracy,
            'test_f1_macro': test_f1_macro,
            'test_f1_micro': test_f1_micro,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm
        }
        
        if test_proba is not None:
            try:
                self.final_results['roc_auc_ovr'] = roc_auc_score(self.y_test, test_proba, multi_class='ovr', average='macro')
            except:
                pass
    
    def _plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Test Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('figs', exist_ok=True)
        plt.savefig('figs/confusion_matrix_classical_classifiers.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save all results to files."""
        print(f"\n--- Saving Results ---")
        
        # Create results directory
        os.makedirs('experiments/results', exist_ok=True)
        
        # Save model comparison
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'model': model_name,
                'val_f1_macro': result['val_f1_macro'],
                'val_accuracy': result['val_accuracy'],
                'best_params': result['params']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('experiments/results/model_comparison.csv', index=False)
        print(f"Model comparison saved to: experiments/results/model_comparison.csv")
        
        # Save final results
        final_results_data = {
            'best_model': self.best_model_name,
            'test_metrics': self.final_results,
            'class_names': self.class_names,
            'num_classes': self.num_classes
        }
        
        with open('experiments/results/final_results.json', 'w') as f:
            json.dump(final_results_data, f, indent=2, default=str)
        print(f"Final results saved to: experiments/results/final_results.json")
        
        print(f"\n=== Training Complete ===")
        print(f"Best model: {self.best_model_name}")
        print(f"Test accuracy: {self.final_results['test_accuracy']:.4f}")
        print(f"Test F1-macro: {self.final_results['test_f1_macro']:.4f}")


def main():
    """Main function to train classical classifiers."""
    print("=== Training Classical Classifiers on MobileNetV2 Features ===\n")
    
    # Setup paths
    features_dir = "data/brainmri_4c/features/mobilenetv2"
    labels_path = "data/brainmri_4c/labels.json"
    
    # Load labels info
    with open(labels_path, 'r') as f:
        labels_info = json.load(f)
    
    # Initialize trainer
    trainer = ClassicalClassifierTrainer(features_dir, labels_info)
    
    # Train all models
    trainer.train_logistic_regression()
    trainer.train_svm_linear()
    trainer.train_svm_rbf()
    
    # Select best model
    trainer.select_best_model()
    
    # Final refit and evaluation
    trainer.final_refit_and_evaluate()
    
    # Save results
    trainer.save_results()


if __name__ == "__main__":
    main()
