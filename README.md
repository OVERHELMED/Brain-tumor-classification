# Reproducible Medical Imaging Project

This project implements best practices for reproducibility in medical imaging research, following established reproducibility checklists and guidelines.

## Project Structure

```
├── data/                    # Data storage
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── src/                    # Source code
│   ├── reproducibility.py # Reproducibility utilities
│   └── train.py           # Main training script
├── configs/                # Configuration files
│   └── config.yaml        # Main configuration
├── experiments/            # Experiment artifacts
│   ├── models/            # Saved models
│   ├── logs/              # Training logs
│   └── checkpoints/       # Model checkpoints
├── figs/                   # Generated figures
├── requirements.txt        # Pinned dependencies
└── README.md              # This file
```

## Reproducibility Practices

### 1. Fixed Random Seeds
- **Global seed**: 42 (configurable in `configs/config.yaml`)
- **Framework-specific seeds**: Separate seeds for PyTorch, NumPy, and CUDA
- **Environment variables**: `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` set for deterministic behavior

### 2. Pinned Dependencies
All dependencies are locked to specific versions in `requirements.txt`:
- **PyTorch**: 2.1.0 (with CUDA support)
- **NumPy**: 1.24.3
- **scikit-learn**: 1.3.2
- **timm**: 0.9.12 (for model architectures)
- **albumentations**: 1.3.1 (for data augmentation)
- **Additional tools**: matplotlib, seaborn, pandas, tqdm, pyyaml

### 3. Deterministic Behavior
- **PyTorch deterministic algorithms**: Enabled where possible
- **CuDNN deterministic**: Enabled for consistent GPU operations
- **CuDNN benchmark**: Disabled to ensure deterministic behavior
- **Mixed precision**: Configurable with proper scaling

### 4. Configuration Management
- **YAML-based configuration**: All hyperparameters in `configs/config.yaml`
- **Version control**: Configuration files tracked in Git
- **Documentation**: All parameters documented with descriptions

### 5. Environment Setup
The reproducibility utilities (`src/reproducibility.py`) provide:
- Automatic seed setting across all random number generators
- Device detection and configuration
- Mixed precision setup
- System information logging

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Edit `configs/config.yaml` to adjust:
- Random seeds
- Model architecture
- Training hyperparameters
- Data paths
- Hardware settings

### 3. Training
```bash
# Run training with reproducibility settings
python src/train.py
```

## Reproducibility Checklist

### ✅ Code and Data
- [x] Source code provided and version controlled
- [x] Configuration files included
- [x] Data preprocessing pipeline documented
- [x] Model architecture clearly defined

### ✅ Environment
- [x] Dependencies pinned to specific versions
- [x] Python version specified
- [x] Hardware requirements documented
- [x] Installation instructions provided

### ✅ Randomness
- [x] Random seeds fixed and documented
- [x] Deterministic algorithms enabled
- [x] Non-deterministic operations minimized
- [x] Seed values reported in results

### ✅ Experiments
- [x] Hyperparameters documented
- [x] Training procedure specified
- [x] Evaluation metrics defined
- [x] Results reproducible across runs

## Medical Imaging Specific Considerations

### Data Handling
- **DICOM compliance**: Ensure proper handling of medical image formats
- **Privacy**: Implement appropriate data anonymization
- **Validation**: Cross-validation strategies for medical datasets
- **Augmentation**: Medical-image appropriate augmentation techniques

### Model Evaluation
- **Clinical metrics**: Sensitivity, specificity, AUC
- **Statistical significance**: Proper statistical testing
- **Cross-validation**: Stratified splits for medical data
- **External validation**: When possible, test on external datasets

### Documentation
- **Clinical context**: Document medical relevance
- **Limitations**: Clearly state model limitations
- **Ethical considerations**: Address bias and fairness
- **Regulatory compliance**: Consider relevant medical device regulations

## Citation

If you use this reproducibility framework in your research, please cite:

```bibtex
@software{reproducible_medical_imaging,
  title={Reproducible Medical Imaging Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo},
  note={Following medical imaging reproducibility best practices}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## Acknowledgments

This framework follows reproducibility guidelines from:
- [Machine Learning Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- [Medical Image Analysis Reproducibility Guidelines](https://www.nature.com/articles/s41591-021-01614-0)
- [PyTorch Reproducibility Best Practices](https://pytorch.org/docs/stable/notes/randomness.html)
