"""
Extract MobileNetV2 embeddings for Brain MRI dataset.

This script loads a pretrained MobileNetV2 model, applies preprocessing,
and extracts penultimate-layer features for train/val/test splits.
Features are saved as .npy arrays for use with lightweight classifiers.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FeatureExtractionDataset(Dataset):
    """
    Dataset for feature extraction with proper preprocessing.
    """
    
    def __init__(self, csv_path: str, transform: A.Compose = None):
        """
        Initialize the dataset.
        
        Args:
            csv_path (str): Path to CSV file with image paths and labels
            transform (A.Compose): Albumentations transform pipeline
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple[torch.Tensor, int]: (image_tensor, label)
        """
        row = self.df.iloc[idx]
        image_path = row['path']
        label = int(row['label'])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Fallback transform
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        return image, label


def get_preprocessing_transforms() -> A.Compose:
    """
    Get preprocessing transforms for feature extraction.
    
    Returns:
        A.Compose: Albumentations transform pipeline
    """
    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return transform


def create_mobilenetv2_model() -> nn.Module:
    """
    Create MobileNetV2 model for feature extraction.
    
    Returns:
        nn.Module: MobileNetV2 model without classifier
    """
    # Create MobileNetV2 with no classifier for feature extraction
    model = timm.create_model(
        'mobilenetv2_100',
        pretrained=True,
        num_classes=0,  # Remove classifier
        global_pool='avg'  # Global average pooling for embeddings
    )
    
    # Set to evaluation mode
    model.eval()
    
    # Print model info
    print(f"Created MobileNetV2 model:")
    print(f"  - Architecture: mobilenetv2_100")
    print(f"  - Pretrained: True")
    print(f"  - Global pooling: avg")
    print(f"  - Number of classes: 0 (feature extraction mode)")
    
    return model


def extract_features(model: nn.Module, 
                    dataloader: DataLoader, 
                    device: torch.device,
                    split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from a model for a given dataloader.
    
    Args:
        model (nn.Module): Model to extract features from
        dataloader (DataLoader): Data loader for the split
        device (torch.device): Device to run inference on
        split_name (str): Name of the split (for logging)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (features, labels)
    """
    features_list = []
    labels_list = []
    
    model = model.to(device)
    
    print(f"\nExtracting features for {split_name} split...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Extracting {split_name}")):
            images = images.to(device)
            
            # Extract features
            features = model(images)
            
            # Move to CPU and convert to numpy
            features_np = features.cpu().numpy()
            labels_np = labels.numpy()
            
            features_list.append(features_np)
            labels_list.append(labels_np)
    
    # Concatenate all batches
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    
    print(f"  Extracted {X.shape[0]} samples")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Labels shape: {y.shape}")
    
    return X, y


def save_features(X: np.ndarray, y: np.ndarray, split_name: str, output_dir: str) -> None:
    """
    Save features and labels to .npy files.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Label vector
        split_name (str): Name of the split
        output_dir (str): Output directory
    """
    X_path = os.path.join(output_dir, f"{split_name}_X.npy")
    y_path = os.path.join(output_dir, f"{split_name}_y.npy")
    
    np.save(X_path, X)
    np.save(y_path, y)
    
    print(f"  Saved features to: {X_path}")
    print(f"  Saved labels to: {y_path}")


def main():
    """
    Main function to extract MobileNetV2 features for all splits.
    """
    print("=== MobileNetV2 Feature Extraction for Brain MRI Dataset ===\n")
    
    # Setup paths
    base_dir = "data/brainmri_4c"
    features_dir = os.path.join(base_dir, "features", "mobilenetv2")
    
    # Create features directory
    os.makedirs(features_dir, exist_ok=True)
    
    # Load labels mapping
    labels_path = os.path.join(base_dir, "labels.json")
    with open(labels_path, 'r') as f:
        labels_info = json.load(f)
    
    print(f"Loaded labels mapping: {labels_info['class_to_label']}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print(f"\n--- Creating MobileNetV2 Model ---")
    model = create_mobilenetv2_model()
    
    # Get preprocessing transforms
    transform = get_preprocessing_transforms()
    
    # Create datasets and data loaders
    print(f"\n--- Creating Datasets and Data Loaders ---")
    
    splits = ['train', 'val', 'test']
    dataloaders = {}
    
    for split in splits:
        csv_path = os.path.join(base_dir, f"{split}.csv")
        dataset = FeatureExtractionDataset(csv_path, transform)
        
        dataloader = DataLoader(
            dataset,
            batch_size=32,  # Small batch size for laptop
            shuffle=False,  # No need to shuffle for feature extraction
            num_workers=2,  # Reduced for stability
            pin_memory=True
        )
        
        dataloaders[split] = dataloader
        print(f"Created {split} dataloader: {len(dataloader)} batches")
    
    # Extract features for each split
    print(f"\n--- Extracting Features ---")
    
    for split in splits:
        X, y = extract_features(model, dataloaders[split], device, split)
        save_features(X, y, split, features_dir)
        
        # Sanity check
        print(f"  {split.capitalize()} split:")
        print(f"    Features shape: {X.shape}")
        print(f"    Labels shape: {y.shape}")
        print(f"    Feature dimension: {X.shape[1]} (expected ~1280 for MobileNetV2)")
        print(f"    Unique labels: {np.unique(y)}")
    
    # Final summary
    print(f"\n=== Feature Extraction Complete ===")
    print(f"Features saved to: {features_dir}")
    
    # List all created files
    feature_files = os.listdir(features_dir)
    print(f"\nCreated files:")
    for file in sorted(feature_files):
        file_path = os.path.join(features_dir, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  {file} ({file_size:.1f} MB)")
    
    print(f"\nReady for Step 3: Train classical classifiers on extracted features!")


if __name__ == "__main__":
    main()
