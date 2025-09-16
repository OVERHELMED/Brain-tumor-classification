# Reproducibility Section for Paper

## Reproducibility Practices

To ensure the reproducibility of our results and enable fair comparison with future work, we implemented comprehensive reproducibility practices following established medical imaging reproducibility checklists [1,2]. Our approach addresses all key aspects of reproducible research in medical image analysis.

### Fixed Random Seeds and Deterministic Behavior

We set fixed random seeds across all random number generators to ensure consistent initialization and training behavior:

- **Global seed**: 42 (configurable)
- **Framework-specific seeds**: Separate seeds for PyTorch (42), NumPy (42), and CUDA (42)
- **Environment variables**: Set `PYTHONHASHSEED=42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic behavior
- **Deterministic algorithms**: Enabled PyTorch's deterministic algorithms where possible using `torch.use_deterministic_algorithms(True)`
- **CuDNN settings**: Set `torch.backends.cudnn.deterministic=True` and `torch.backends.cudnn.benchmark=False` for consistent GPU operations

### Pinned Dependencies and Environment

All dependencies are locked to specific versions to ensure consistent behavior across different environments:

- **PyTorch**: 2.1.0 (with CUDA 11.8 support)
- **NumPy**: 1.24.3
- **scikit-learn**: 1.3.2
- **timm**: 0.9.12 (for model architectures)
- **albumentations**: 1.3.1 (for data augmentation)
- **Additional dependencies**: matplotlib (3.7.2), seaborn (0.12.2), pandas (2.0.3), tqdm (4.66.1)

The complete list of dependencies with exact versions is provided in `requirements.txt`.

### Configuration Management

All hyperparameters and experimental settings are stored in version-controlled YAML configuration files (`configs/config.yaml`), including:

- Model architecture and training parameters
- Data preprocessing and augmentation settings
- Hardware configuration (device, mixed precision)
- Path configurations for data and outputs
- Experiment metadata and tags

This approach ensures that all experimental parameters are documented and reproducible.

### Code and Data Availability

- **Source code**: Complete implementation provided with detailed documentation
- **Configuration files**: All experimental configurations version-controlled
- **Reproducibility utilities**: Custom module (`src/reproducibility.py`) for consistent environment setup
- **Installation instructions**: Step-by-step setup guide in README.md
- **System requirements**: Hardware and software requirements documented

### Medical Imaging Specific Considerations

Following medical imaging reproducibility best practices:

- **Data preprocessing**: Standardized preprocessing pipeline with documented parameters
- **Cross-validation**: Stratified splits respecting medical data characteristics
- **Statistical evaluation**: Proper statistical testing and confidence intervals
- **Clinical metrics**: Reporting of clinically relevant metrics (sensitivity, specificity, AUC)
- **Bias considerations**: Documentation of potential sources of bias in data and methodology

### Validation of Reproducibility

We validated our reproducibility setup by:

1. Running identical experiments multiple times with the same configuration
2. Verifying that random number generation is deterministic across runs
3. Confirming that model training produces identical results when using the same seeds
4. Testing across different hardware configurations (CPU/GPU)

All experiments can be reproduced by following the provided setup instructions and using the exact dependency versions specified.

### Compliance with Reproducibility Guidelines

Our approach follows established reproducibility checklists:

- ✅ **Machine Learning Reproducibility Checklist** [1]: Fixed seeds, pinned dependencies, documented hyperparameters
- ✅ **Medical Image Analysis Guidelines** [2]: Clinical metrics, statistical validation, bias documentation
- ✅ **PyTorch Best Practices** [3]: Deterministic algorithms, proper seed management

This comprehensive approach ensures that our results can be reliably reproduced and that fair comparisons can be made with future work in the field.

---

**References:**

[1] Pineau, J., et al. "Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program)." *Journal of Machine Learning Research* 22.164 (2021): 1-20.

[2] Maier-Hein, L., et al. "Why rankings of biomedical image analysis competitions should be interpreted with care." *Nature communications* 9.1 (2018): 5217.

[3] PyTorch Documentation. "Reproducibility." https://pytorch.org/docs/stable/notes/randomness.html
