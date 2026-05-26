"""
Data preprocessing and utilities for trading data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List, Optional
import cv2
from pathlib import Path
from config import DataConfig, ModelConfig


class DataPreprocessor:
    """Handles time series data preprocessing for LSTM models"""
    
    def __init__(self, scaler_type: str = "minmax"):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: "minmax" or "standard"
        """
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.scaler_type = scaler_type
    
    def normalize_data(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Normalize data using fitted scaler
        
        Args:
            data: Array of shape (N, features)
            
        Returns:
            normalized_data, scaler_params
        """
        # Reshape to 2D if needed
        original_shape = data.shape
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        normalized = self.scaler.fit_transform(data)
        
        scaler_params = {
            "type": self.scaler_type,
            "scale": getattr(self.scaler, 'scale_', None),
            "min": getattr(self.scaler, 'data_min_', None),
            "max": getattr(self.scaler, 'data_max_', None),
            "mean": getattr(self.scaler, 'mean_', None),
            "var": getattr(self.scaler, 'var_', None),
        }
        
        return normalized, scaler_params
    
    def denormalize_data(self, normalized_data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data"""
        return self.scaler.inverse_transform(normalized_data)
    
    def create_sequences(self, data: np.ndarray, seq_length: int, 
                        target_col: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            data: Array of shape (N, features)
            seq_length: Length of input sequences
            target_col: Column index for target variable
            
        Returns:
            X, y arrays of sequences
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, target_col])
        
        return np.array(X), np.array(y)
    
    def process_ohlcv_data(self, df: pd.DataFrame, 
                          seq_length: int = DataConfig.SEQ_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process OHLCV data from dataframe
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            seq_length: Sequence length for LSTM
            
        Returns:
            X, y sequences
        """
        # Extract OHLCV columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        data = df[cols].values
        
        # Normalize
        normalized, _ = self.normalize_data(data)
        
        # Create sequences
        X, y = self.create_sequences(normalized, seq_length, target_col=3)  # close is column 3
        
        return X, y


class ImagePreprocessor:
    """Handles image preprocessing for chart analysis"""
    
    @staticmethod
    def load_and_preprocess_image(image_path: str, 
                                  target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Load and preprocess an image
        
        Args:
            image_path: Path to image file
            target_size: Target image size (height, width)
            
        Returns:
            Preprocessed image array
        """
        # Load image
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    @staticmethod
    def preprocess_image_batch(images: List[str], 
                               target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess batch of images
        
        Args:
            images: List of image paths
            target_size: Target image size
            
        Returns:
            Batch of preprocessed images
        """
        batch = []
        for img_path in images:
            img = ImagePreprocessor.load_and_preprocess_image(img_path, target_size)
            batch.append(img)
        
        return np.array(batch)
    
    @staticmethod
    def augment_image(image: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation to image
        
        Args:
            image: Image array
            
        Returns:
            List of augmented images
        """
        augmented = [image]
        
        # Horizontal flip
        augmented.append(np.fliplr(image))
        
        # Slight rotation (via warping)
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        angle = 15
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        augmented.append(rotated)
        
        return augmented


class DataLoader:
    """Create PyTorch/TensorFlow compatible data loaders"""
    
    @staticmethod
    def generate_synthetic_ohlcv_data(num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate random walk price data
        np.random.seed(42)
        close_prices = np.cumsum(np.random.randn(num_samples)) + 100
        
        # Generate OHLCV
        data = {
            'open': close_prices + np.random.randn(num_samples) * 0.5,
            'high': close_prices + np.abs(np.random.randn(num_samples)) * 1.5,
            'low': close_prices - np.abs(np.random.randn(num_samples)) * 1.5,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, num_samples),
        }
        
        df = pd.DataFrame(data)
        return df.astype(np.float32)
    
    @staticmethod
    def train_val_test_split(X: np.ndarray, y: np.ndarray, 
                            train_ratio: float = 0.7, 
                            val_ratio: float = 0.15) -> Tuple[
                                Tuple[np.ndarray, np.ndarray],
                                Tuple[np.ndarray, np.ndarray],
                                Tuple[np.ndarray, np.ndarray]
                            ]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Input sequences
            y: Target values
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
