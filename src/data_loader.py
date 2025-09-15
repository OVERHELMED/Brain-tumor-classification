"""
Data loading utilities for Brain MRI 4-class tumor classification dataset.

This module provides data loading functionality for the brain MRI dataset
with proper preprocessing and augmentation for medical imaging.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from typing import Tuple, Dict, List, Optional
import yaml


class BrainMRIDataset(Dataset):
    """
    Dataset class for Brain MRI 4-class tumor classification.
    
    Classes:
    - glioma
    - meningioma  
    - notumor
    - pituitary
    """
    
    def __init__(self, 
                 data_dir: str,
                 class_names: List[str],
                 transform: Optional[A.Compose] = None,
                 split: str = "train"):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Path to the data directory
            class_names (List[str]): List of class names
            transform (Optional[A.Compose]): Albumentations transform pipeline
            split (str): Data split ('train', 'test')
        """
        self.data_dir = data_dir
        self.class_names = class_names
        self.transform = transform
        self.split = split
        
        # Create class to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        
        # Load data samples
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Load all samples from the data directory.
        
        Returns:
            List[Tuple[str, int]]: List of (image_path, class_index) tuples
        """
        samples = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} does not exist")
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in the class directory
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                samples.append((image_path, class_idx))
        
        return samples
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            Dict[str, int]: Dictionary mapping class names to counts
        """
        distribution = {class_name: 0 for class_name in self.class_names}
        
        for _, class_idx in self.samples:
            class_name = self.class_names[class_idx]
            distribution[class_name] += 1
        
        return distribution
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple[torch.Tensor, int]: (image_tensor, class_index)
        """
        image_path, class_idx = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default transform: resize and convert to tensor
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        return image, class_idx


def get_augmentation_transforms(config: Dict) -> Dict[str, A.Compose]:
    """
    Get augmentation transforms for training and validation.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Dict[str, A.Compose]: Dictionary containing train and val transforms
    """
    image_size = config['data']['image_size']
    
    # Training transforms with augmentation
    train_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=config['augmentation']['horizontal_flip']),
        A.VerticalFlip(p=config['augmentation']['vertical_flip']),
        A.Rotate(limit=config['augmentation']['rotation_limit'], p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=config['augmentation']['brightness_contrast'],
            contrast_limit=config['augmentation']['brightness_contrast'],
            p=0.5
        ),
        A.GaussNoise(
            var_limit=(10.0, config['augmentation']['gaussian_noise'] * 255),
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # Validation transforms (no augmentation)
    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def create_data_loaders(config: Dict) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Dict[str, DataLoader]: Dictionary containing data loaders
    """
    # Get transforms
    transforms_dict = get_augmentation_transforms(config)
    
    # Create datasets
    datasets = {}
    
    # Training dataset
    train_dataset = BrainMRIDataset(
        data_dir=config['paths']['train_dir'],
        class_names=config['data']['classes'],
        transform=transforms_dict['train'],
        split='train'
    )
    
    # Test dataset
    test_dataset = BrainMRIDataset(
        data_dir=config['paths']['test_dir'],
        class_names=config['data']['classes'],
        transform=transforms_dict['test'],
        split='test'
    )
    
    # Create validation dataset from training data
    # Split training data into train/val
    train_size = int(config['data']['train_split'] * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seeds']['global_seed'])
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = transforms_dict['val']
    
    datasets['train'] = train_dataset
    datasets['val'] = val_dataset
    datasets['test'] = test_dataset
    
    # Create data loaders
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        shuffle = (split == 'train')
        data_loaders[split] = DataLoader(
            datasets[split],
            batch_size=config['data']['batch_size'],
            shuffle=shuffle,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            persistent_workers=True if config['data']['num_workers'] > 0 else False
        )
    
    return data_loaders


def get_class_weights(config: Dict, train_dataset: BrainMRIDataset) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        config (Dict): Configuration dictionary
        train_dataset (BrainMRIDataset): Training dataset
        
    Returns:
        torch.Tensor: Class weights tensor
    """
    class_counts = list(train_dataset._get_class_distribution().values())
    total_samples = sum(class_counts)
    
    # Calculate weights (inverse frequency)
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    
    return torch.FloatTensor(class_weights)


if __name__ == "__main__":
    # Example usage
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    data_loaders = create_data_loaders(config)
    
    # Print dataset information
    for split, loader in data_loaders.items():
        print(f"{split.capitalize()} DataLoader:")
        print(f"  - Number of batches: {len(loader)}")
        print(f"  - Batch size: {loader.batch_size}")
        print(f"  - Total samples: {len(loader.dataset)}")
        
        # Get a sample batch
        for batch_idx, (images, labels) in enumerate(loader):
            print(f"  - Sample batch shape: {images.shape}")
            print(f"  - Sample labels: {labels}")
            break
        print()
