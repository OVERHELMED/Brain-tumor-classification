"""
Main training script with reproducibility features.

This script demonstrates how to use the reproducibility utilities
and configuration system for consistent results.
"""

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from reproducibility import set_reproducible_environment, get_device, setup_mixed_precision, log_system_info


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_reproducible_training(config: dict) -> tuple:
    """
    Setup reproducible training environment based on configuration.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        tuple: (device, scaler, config)
    """
    # Set reproducible environment
    set_reproducible_environment(
        seed=config['seeds']['global_seed'],
        deterministic=config['hardware']['deterministic']
    )
    
    # Get device
    device = get_device(config['hardware']['device'])
    
    # Setup mixed precision if enabled
    scaler = None
    if config['hardware']['mixed_precision']:
        scaler = setup_mixed_precision()
    
    # Log system information
    log_system_info()
    
    return device, scaler, config


def create_model(config: dict, device: torch.device) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config (dict): Configuration dictionary.
        device (torch.device): Device to place model on.
        
    Returns:
        nn.Module: Created model.
    """
    import timm
    
    model = timm.create_model(
        config['model']['name'],
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes']
    )
    
    model = model.to(device)
    return model


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model (nn.Module): Model to optimize.
        config (dict): Configuration dictionary.
        
    Returns:
        optim.Optimizer: Created optimizer.
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: dict) -> Optional[object]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer (optim.Optimizer): Optimizer to schedule.
        config (dict): Configuration dictionary.
        
    Returns:
        Optional[object]: Created scheduler or None.
    """
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['training']['epochs']
        )
        return scheduler
    elif config['training']['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=30, 
            gamma=0.1
        )
        return scheduler
    else:
        return None


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device, scaler=None) -> float:
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): Training data loader.
        optimizer (optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use.
        scaler: Mixed precision scaler (optional).
        
    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
             device: torch.device) -> tuple:
    """
    Validate model.
    
    Args:
        model (nn.Module): Model to validate.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use.
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return total_loss / len(dataloader), accuracy


def main():
    """
    Main training function.
    """
    # Load configuration
    config = load_config()
    
    # Setup reproducible environment
    device, scaler, config = setup_reproducible_training(config)
    
    # Create model
    model = create_model(config, device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model: {config['model']['name']}")
    print(f"Training for {config['training']['epochs']} epochs")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print("-" * 50)
    
    # Training loop would go here
    # For demonstration, we'll just print the setup
    print("Training setup completed successfully!")
    print("Ready for data loading and training loop implementation.")


if __name__ == "__main__":
    main()
