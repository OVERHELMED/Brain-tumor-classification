"""
Create CSV indexes for Brain MRI dataset with stratified train/val/test splits.

This script scans the existing Training and Testing folders, creates a stratified
validation split from Training, and saves three CSVs with columns: path, label, split.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import glob


def get_class_mapping() -> Dict[str, int]:
    """
    Define the mapping from class folder names to numeric labels.
    
    Returns:
        Dict[str, int]: Mapping from class name to label index
    """
    return {
        'glioma': 0,
        'meningioma': 1, 
        'pituitary': 2,
        'notumor': 3
    }


def scan_folder_for_images(folder_path: str, class_name: str, class_label: int) -> List[Tuple[str, int]]:
    """
    Scan a class folder for all image files.
    
    Args:
        folder_path (str): Path to the class folder
        class_name (str): Name of the class
        class_label (int): Numeric label for the class
        
    Returns:
        List[Tuple[str, int]]: List of (image_path, label) tuples
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF']
    
    samples = []
    
    for extension in image_extensions:
        pattern = os.path.join(folder_path, extension)
        image_files = glob.glob(pattern)
        
        for image_file in image_files:
            # Convert to relative path from project root
            relative_path = os.path.relpath(image_file, start=os.getcwd())
            # Normalize path separators for cross-platform compatibility
            relative_path = relative_path.replace('\\', '/')
            samples.append((relative_path, class_label))
    
    return samples


def create_training_dataframe(data_dir: str, class_mapping: Dict[str, int]) -> pd.DataFrame:
    """
    Create DataFrame from Training folder with all images and labels.
    
    Args:
        data_dir (str): Path to the training data directory
        class_mapping (Dict[str, int]): Mapping from class names to labels
        
    Returns:
        pd.DataFrame: DataFrame with columns ['path', 'label']
    """
    all_samples = []
    
    for class_name, class_label in class_mapping.items():
        class_folder = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_folder):
            print(f"Warning: Class folder {class_folder} does not exist")
            continue
            
        print(f"Scanning {class_name} folder...")
        samples = scan_folder_for_images(class_folder, class_name, class_label)
        all_samples.extend(samples)
        print(f"  Found {len(samples)} images")
    
    # Create DataFrame
    df = pd.DataFrame(all_samples, columns=['path', 'label'])
    
    print(f"\nTraining dataset summary:")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:")
    for class_name, class_label in class_mapping.items():
        count = (df['label'] == class_label).sum()
        print(f"  {class_name} (label {class_label}): {count} samples")
    
    return df


def create_testing_dataframe(data_dir: str, class_mapping: Dict[str, int]) -> pd.DataFrame:
    """
    Create DataFrame from Testing folder with all images and labels.
    
    Args:
        data_dir (str): Path to the testing data directory
        class_mapping (Dict[str, int]): Mapping from class names to labels
        
    Returns:
        pd.DataFrame: DataFrame with columns ['path', 'label']
    """
    all_samples = []
    
    for class_name, class_label in class_mapping.items():
        class_folder = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_folder):
            print(f"Warning: Class folder {class_folder} does not exist")
            continue
            
        print(f"Scanning {class_name} folder...")
        samples = scan_folder_for_images(class_folder, class_name, class_label)
        all_samples.extend(samples)
        print(f"  Found {len(samples)} images")
    
    # Create DataFrame
    df = pd.DataFrame(all_samples, columns=['path', 'label'])
    
    print(f"\nTesting dataset summary:")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:")
    for class_name, class_label in class_mapping.items():
        count = (df['label'] == class_label).sum()
        print(f"  {class_name} (label {class_label}): {count} samples")
    
    return df


def create_stratified_split(df: pd.DataFrame, test_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation split.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'label' column
        test_size (float): Proportion for validation set (default 0.1 = 10%)
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)
    """
    # Stratified split
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['label'], 
        random_state=random_state
    )
    
    print(f"\nStratified split results:")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Split ratio: {len(val_df) / len(df):.2%}")
    
    # Print class distribution for each split
    print(f"\nTrain set class distribution:")
    for class_name, class_label in get_class_mapping().items():
        count = (train_df['label'] == class_label).sum()
        print(f"  {class_name} (label {class_label}): {count} samples")
    
    print(f"\nValidation set class distribution:")
    for class_name, class_label in get_class_mapping().items():
        count = (val_df['label'] == class_label).sum()
        print(f"  {class_name} (label {class_label}): {count} samples")
    
    return train_df, val_df


def save_labels_mapping(class_mapping: Dict[str, int], output_dir: str) -> None:
    """
    Save class mapping to JSON file.
    
    Args:
        class_mapping (Dict[str, int]): Class name to label mapping
        output_dir (str): Output directory path
    """
    labels_path = os.path.join(output_dir, 'labels.json')
    
    # Add reverse mapping for convenience
    mapping_data = {
        'class_to_label': class_mapping,
        'label_to_class': {v: k for k, v in class_mapping.items()},
        'num_classes': len(class_mapping),
        'class_names': list(class_mapping.keys())
    }
    
    with open(labels_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"\nLabels mapping saved to: {labels_path}")


def main():
    """
    Main function to create CSV indexes for the Brain MRI dataset.
    """
    print("=== Creating CSV Indexes for Brain MRI Dataset ===\n")
    
    # Define paths
    base_dir = "data/brainmri_4c"
    training_dir = os.path.join(base_dir, "training")
    testing_dir = os.path.join(base_dir, "testing")
    output_dir = base_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class mapping
    class_mapping = get_class_mapping()
    print(f"Class mapping: {class_mapping}")
    
    # Create training DataFrame
    print(f"\n--- Scanning Training Folder ---")
    train_full_df = create_training_dataframe(training_dir, class_mapping)
    
    # Create stratified split
    print(f"\n--- Creating Stratified Split ---")
    train_df, val_df = create_stratified_split(train_full_df, test_size=0.1, random_state=42)
    
    # Create testing DataFrame
    print(f"\n--- Scanning Testing Folder ---")
    test_df = create_testing_dataframe(testing_dir, class_mapping)
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Save CSV files
    print(f"\n--- Saving CSV Files ---")
    
    train_csv_path = os.path.join(output_dir, 'train.csv')
    val_csv_path = os.path.join(output_dir, 'val.csv')
    test_csv_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"Train CSV saved to: {train_csv_path}")
    print(f"Validation CSV saved to: {val_csv_path}")
    print(f"Test CSV saved to: {test_csv_path}")
    
    # Save labels mapping
    save_labels_mapping(class_mapping, output_dir)
    
    # Print final summary
    print(f"\n=== Final Summary ===")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}")
    
    # Verify file existence
    csv_files = ['train.csv', 'val.csv', 'test.csv', 'labels.json']
    print(f"\nFiles created in {output_dir}:")
    for file_name in csv_files:
        file_path = os.path.join(output_dir, file_name)
        exists = os.path.exists(file_path)
        print(f"  {file_name}: {'✓' if exists else '✗'}")
    
    print(f"\n=== Step 1 Complete: CSV Indexes Created ===")
    print("Ready for Step 2: MobileNetV2 feature extraction")


if __name__ == "__main__":
    main()
