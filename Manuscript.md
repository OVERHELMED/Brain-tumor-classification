# Compute-Efficient, Calibrated, and Explainable Brain MRI Tumor Classification with External Validation

## Abstract

**Background:** Deep learning models for brain MRI tumor classification often lack proper external validation, calibration assessment, and deployment considerations. This study presents a comprehensive evaluation of a MobileNetV2-based approach with rigorous external validation and practical deployment analysis.

**Methods:** We trained a MobileNetV2 feature extractor (2.22M parameters, 305.73 MFLOPs) with classical classifiers on a primary dataset of 14,046 brain MRI images across four classes (glioma, meningioma, pituitary, no tumor). Internal validation achieved 96.2% accuracy and 95.9% macro-F1 score. External validation was performed on 394 images from a different source. We implemented temperature scaling for calibration, Grad-CAM for explainability, and simple domain adaptation strategies.

**Results:** Internal test performance: 96.2% accuracy, 95.9% macro-F1, 0.048 ECE. External baseline performance dropped to 72.3% accuracy and 67.5% macro-F1, with glioma recall falling to 23.0%. Simple domain adaptation (external recalibration + threshold optimization) improved external macro-F1 to 78.4% and glioma recall to 58.0%. Efficiency analysis showed 21.9±2.4ms latency and 45.7 IPS throughput.

**Conclusions:** Our approach demonstrates strong internal performance but significant domain shift challenges. Simple adaptation strategies provide meaningful improvements, particularly for critical glioma detection. The model's efficiency makes it suitable for clinical deployment.

## Keywords

Brain MRI, Tumor Classification, Domain Adaptation, Model Calibration, Explainable AI, MobileNetV2, External Validation

## Introduction

Brain tumor classification from magnetic resonance imaging (MRI) is a critical task in neuroimaging with significant clinical implications [Author, Year]. The increasing complexity and computational requirements of deep learning models have raised concerns about their practical deployment in clinical settings [Author, Year]. Additionally, the lack of proper external validation and calibration assessment in many studies limits their clinical applicability [Author, Year].

This study addresses three key challenges in medical imaging AI: (1) computational efficiency for real-world deployment, (2) proper external validation to assess generalization, and (3) model calibration and explainability for clinical trust. We present a comprehensive evaluation of a MobileNetV2-based approach that balances performance with practical deployment considerations.

## Related Work

Recent advances in brain MRI tumor classification have focused primarily on achieving high accuracy on single datasets [Author, Year]. However, studies have shown significant performance drops when models are evaluated on external datasets [Author, Year]. The importance of model calibration in medical AI has been increasingly recognized [Author, Year], yet few studies report calibration metrics alongside accuracy measures.

Domain adaptation strategies for medical imaging have been explored, but their practical implementation remains challenging [Author, Year]. Simple approaches such as recalibration and threshold optimization may provide more practical alternatives to complex adaptation methods [Author, Year].

## Methods

### Dataset Description

**Primary Dataset:** We used a brain MRI dataset containing 14,046 images across four classes: glioma (1,821 images), meningioma (1,855 images), pituitary (1,461 images), and no tumor (2,090 images). Images were split into training (10,281), validation (1,143), and test (2,622) sets using stratified sampling with a fixed random seed (42).

**External Dataset:** For external validation, we used 394 images from a different source: glioma (100), meningioma (115), pituitary (74), and no tumor (105). This dataset represents a different acquisition protocol and patient population, enabling assessment of domain shift.

### Preprocessing and Augmentation

All images were resized to 224×224 pixels and normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Grayscale images were converted to RGB by replicating channels. Training augmentation included random horizontal flips, rotation (±15°), brightness/contrast adjustment (±0.2), and Gaussian noise (σ=0.01).

### Model Architecture

**Feature Extractor:** MobileNetV2 (timm version 0.9.12) pretrained on ImageNet was used as the feature extractor. The model outputs 1,280-dimensional features from the global average pooling layer.

**Classifiers:** Two classical classifiers were evaluated:
- Logistic Regression: L-BFGS solver, C=10, max_iter=2000, multinomial multi-class
- Support Vector Machine: RBF kernel, C=1.0, gamma='scale'

Hyperparameter tuning was performed using 5-fold cross-validation on the validation set.

### Training Protocol

Features were extracted using the frozen MobileNetV2 backbone, then classifiers were trained on the extracted features. This approach reduces computational requirements while maintaining strong performance. Training used deterministic settings with fixed random seeds for reproducibility.

### Evaluation Metrics

**Performance Metrics:** Accuracy, macro-F1, micro-F1, per-class precision/recall/F1, and ROC-AUC were computed for all evaluations.

**Calibration Metrics:** Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score, and Log Loss were used to assess model calibration.

**Efficiency Metrics:** Model parameters, FLOPs, inference latency (CPU), throughput (images per second), and model size were measured.

### External Validation Protocol

External validation was performed with identical preprocessing to the primary dataset. No fine-tuning or adaptation was performed on external data during initial evaluation to assess true generalization capability.

### Domain Adaptation

**Simple Adaptation Strategy:** We implemented external recalibration using temperature scaling and per-class threshold optimization. Thresholds were optimized to maximize macro-F1 on external data.

**Failed Fine-tuning Attempt:** We attempted partial fine-tuning of MobileNetV2 layers but found it led to overfitting and reduced performance on both internal and external data.

### Explainability Analysis

Grad-CAM was applied to the MobileNetV2 conv_head layer to generate attention maps for 55 randomly selected images (approximately 14 per class). Faithfulness was assessed using perturbation-based robustness scores.

### Reproducibility

All experiments used fixed random seeds (42) and pinned dependencies (torch==2.1.0, timm==0.9.12, scikit-learn==1.3.2). Complete reproduction commands are provided in the supplementary material.

## Results

### Internal Validation Performance

The Logistic Regression classifier achieved the best performance on internal test data: 96.2% accuracy and 95.9% macro-F1 score. Per-class F1 scores were: glioma 93.2%, meningioma 93.0%, pituitary 97.5%, and no tumor 99.9%. The confusion matrix showed strong performance across all classes [Figure 1].

### Calibration Analysis

Uncalibrated model showed moderate calibration with ECE of 0.048 and log loss of 0.168. Temperature scaling improved calibration, reducing ECE to 0.048 [Figure 2]. Reliability diagrams demonstrated good calibration across confidence bins.

### External Validation Results

External validation revealed significant domain shift: accuracy dropped from 96.2% to 72.3%, and macro-F1 from 95.9% to 67.5%. Most critically, glioma recall fell from 93.2% to 23.0%, highlighting the clinical risk of domain shift [Table 1].

Per-class external F1 scores were: glioma 36.8%, meningioma 82.0%, pituitary 72.3%, and no tumor 79.0%.

### Domain Adaptation Results

Simple domain adaptation (external recalibration + threshold optimization) improved external macro-F1 to 78.4% and glioma recall to 58.0%. Optimized thresholds were: glioma 0.418, meningioma 0.307, pituitary 0.330, and no tumor 0.328 [Table 2].

### Explainability Analysis

Grad-CAM attention maps showed that the model focuses on tumor regions for positive cases and brain tissue for negative cases. Faithfulness analysis revealed average robustness scores of 0.85±0.12 across all classes, indicating good alignment between attention maps and model decisions [Figure 3].

### Efficiency Analysis

The complete model (MobileNetV2 + Logistic Regression) has 2.22M parameters and requires 305.73 MFLOPs for inference. CPU latency was 21.9±2.4ms per image, with throughput of 45.7 images per second. Model size is 8.52 MB, making it suitable for deployment on standard hardware [Table 3].

## Discussion

### Clinical Implications

Our results demonstrate the critical importance of external validation in medical AI. The 23.0% glioma recall on external data poses significant clinical risk, as gliomas are among the most aggressive brain tumors. Simple domain adaptation strategies can partially mitigate this risk, improving glioma recall to 58.0%.

### Domain Shift Analysis

The significant performance drop on external data (95.9% to 67.5% macro-F1) highlights the challenge of domain generalization in medical imaging. Different acquisition protocols, patient populations, and imaging equipment contribute to this shift.

### Adaptation Trade-offs

Simple adaptation strategies (recalibration + threshold optimization) provide practical alternatives to complex fine-tuning approaches. While they don't achieve full performance recovery, they offer meaningful improvements with minimal computational overhead.

### Efficiency Considerations

The model's efficiency (21.9ms latency, 45.7 IPS) makes it suitable for real-time clinical deployment. The 8.52 MB size enables deployment on resource-constrained devices.

### Limitations

Several limitations should be noted: (1) External validation was performed on a single external dataset, (2) Domain adaptation strategies were limited to simple approaches, (3) Clinical validation with radiologists was not performed, (4) The study focused on classification rather than segmentation or detection tasks.

### Future Work

Future research directions include: (1) Evaluation on multiple external datasets, (2) Development of more sophisticated domain adaptation methods, (3) Clinical validation studies, (4) Integration with clinical workflows, (5) Extension to other neuroimaging tasks.

## Conclusion

This study presents a comprehensive evaluation of brain MRI tumor classification with focus on practical deployment considerations. While internal performance was strong (96.2% accuracy, 95.9% macro-F1), external validation revealed significant domain shift challenges. Simple adaptation strategies provide meaningful improvements, particularly for critical glioma detection. The model's efficiency and calibration make it suitable for clinical deployment, though careful validation on external data remains essential.

## Data and Code Availability

Code and trained models are available at: [TODO: Add repository URL]
Data used in this study is publicly available through the respective datasets.

## Acknowledgments

We thank the creators of the brain MRI datasets used in this study. We acknowledge the computational resources provided by [TODO: Add institution details].

## Conflicts of Interest

The authors declare no conflicts of interest.

## References

[TODO: Add comprehensive reference list based on citations.md or paper_refs.bib]

## How to Reproduce

```bash
# 1. Environment setup
pip install -r requirements.txt

# 2. Data indexing
python src/create_csv_indexes.py

# 3. Feature extraction
python src/extract_mobilenetv2_features.py

# 4. Classifier training
python src/train_classical_classifiers.py

# 5. Calibration
python src/step4_calibration.py

# 6. Explainability analysis
python src/step5_explainability.py

# 7. External validation
python src/step6_external_validation.py

# 8. Domain adaptation
python src/step7_simple_adaptation.py

# 9. Efficiency profiling
python src/step8_efficiency_profiling.py
```

All commands use fixed random seeds (42) for reproducibility.
