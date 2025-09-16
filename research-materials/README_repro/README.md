# Reproducibility Guide for Brain MRI Tumor Classification Study

## Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Complete Reproduction Pipeline

### Step 1: Data Indexing
```bash
python src/create_csv_indexes.py
```
Creates train/val/test splits with fixed random seed (42).

### Step 2: Feature Extraction
```bash
python src/extract_mobilenetv2_features.py
```
Extracts MobileNetV2 features for all images, saves as .npy files.

### Step 3: Classifier Training
```bash
python src/train_classical_classifiers.py
```
Trains Logistic Regression and SVM with hyperparameter tuning.

### Step 4: Calibration Analysis
```bash
python src/step4_calibration.py
```
Performs temperature scaling and calibration assessment.

### Step 5: Explainability Analysis
```bash
python src/step5_explainability.py
```
Generates Grad-CAM attention maps and faithfulness metrics.

### Step 6: External Validation
```bash
python src/step6_external_validation.py
```
Evaluates model on external dataset without adaptation.

### Step 7: Domain Adaptation
```bash
python src/step7_simple_adaptation.py
```
Implements simple adaptation strategies (recalibration + thresholds).

### Step 8: Efficiency Profiling
```bash
python src/step8_efficiency_profiling.py
```
Measures model complexity, latency, and throughput.

## Expected Results

After running all steps, you should obtain:

- Internal accuracy: ~96.2%
- Internal macro-F1: ~95.9%
- External accuracy: ~72.3% (baseline)
- External accuracy: ~77.9% (adapted)
- Glioma recall: 23% → 58% (improvement)

## Verification

Compare your results with `paper_numbers.json` for verification.

## Troubleshooting

- Ensure all dependencies are installed correctly
- Check that data paths are correct
- Verify random seeds are set to 42
- GPU not required (CPU-only inference)

## Contact

For questions about reproduction, contact: [TODO: Add contact information]
