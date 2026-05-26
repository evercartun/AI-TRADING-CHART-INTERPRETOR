"""
PyTorch Dataset classes for trading data
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from pathlib import Path


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Input sequences of shape (N, seq_length, features)
            y: Target values of shape (N,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # Add feature dimension
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ChartImageDataset(Dataset):
    """PyTorch Dataset for chart images"""
    
    def __init__(self, image_paths: list, labels: np.ndarray, 
                 transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: Array of labels (0=down, 1=neutral, 2=up)
            transform: Optional image transformation
        """
        self.image_paths = image_paths
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        import cv2
        
        # Load image
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        image = torch.FloatTensor(np.transpose(image, (2, 0, 1)))
        label = self.labels[idx]
        
        return image, label


def create_time_series_loaders(X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                batch_size: int = 32,
                                num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for time series data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def create_image_loaders(image_paths: list, labels: np.ndarray,
                         train_indices: np.ndarray,
                         val_indices: np.ndarray,
                         test_indices: np.ndarray,
                         batch_size: int = 32,
                         num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for image data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_images = [image_paths[i] for i in train_indices]
    val_images = [image_paths[i] for i in val_indices]
    test_images = [image_paths[i] for i in test_indices]
    
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]
    
    train_dataset = ChartImageDataset(train_images, train_labels)
    val_dataset = ChartImageDataset(val_images, val_labels)
    test_dataset = ChartImageDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
