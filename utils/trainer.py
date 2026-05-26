"""
Training utilities for PyTorch models
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Dict, Optional
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import ModelConfig, TrainingConfig


class TrainerBase:
    """Base trainer class with common functionality"""
    
    def __init__(self, model: nn.Module, device: str = TrainingConfig.DEVICE):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'best_loss': loss,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.best_loss = checkpoint['best_loss']
        return checkpoint['epoch']
    
    def early_stopping(self, val_loss: float, patience: int = ModelConfig.EARLY_STOPPING_PATIENCE) -> bool:
        """Check if training should stop"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                return True
        return False


class LSTMTrainer(TrainerBase):
    """Trainer for LSTM time series models"""
    
    def __init__(self, model: nn.Module, device: str = TrainingConfig.DEVICE,
                 learning_rate: float = ModelConfig.LEARNING_RATE):
        """Initialize LSTM trainer"""
        super().__init__(model, device)
        
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=ModelConfig.WEIGHT_DECAY
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.loss_fn = nn.MSELoss()
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training"):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.loss_fn(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc="Validating"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.loss_fn(predictions, batch_y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs: int = ModelConfig.EPOCHS,
              checkpoint_dir: str = "checkpoints") -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            if (epoch + 1) % ModelConfig.SAVE_FREQ == 0:
                self.save_checkpoint(
                    f"{checkpoint_dir}/lstm_epoch_{epoch + 1}.pt",
                    epoch,
                    val_loss
                )
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        return self.history


class CNNTrainer(TrainerBase):
    """Trainer for CNN image classification models"""
    
    def __init__(self, model: nn.Module, device: str = TrainingConfig.DEVICE,
                 learning_rate: float = ModelConfig.LEARNING_RATE):
        """Initialize CNN trainer"""
        super().__init__(model, device)
        
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=ModelConfig.WEIGHT_DECAY
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.loss_fn = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch, return loss and accuracy"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training"):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.squeeze().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.loss_fn(logits, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate model, return loss and accuracy"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc="Validating"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.squeeze().to(self.device)
                
                logits = self.model(batch_x)
                loss = self.loss_fn(logits, batch_y)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs: int = ModelConfig.EPOCHS,
              checkpoint_dir: str = "checkpoints") -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            if (epoch + 1) % ModelConfig.SAVE_FREQ == 0:
                self.save_checkpoint(
                    f"{checkpoint_dir}/cnn_epoch_{epoch + 1}.pt",
                    epoch,
                    val_loss
                )
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        return self.history
