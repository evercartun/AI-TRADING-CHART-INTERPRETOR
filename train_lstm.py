"""
Training script for LSTM time series model
Usage: python train_lstm.py
"""

import torch
import numpy as np
from pathlib import Path

from config import ModelConfig, DataConfig
from models.pytorch_models import LSTMPredictor
from utils.preprocessing import DataPreprocessor, DataLoader
from utils.torch_data import create_time_series_loaders
from utils.trainer import LSTMTrainer


def train_lstm_model():
    """Train LSTM model on synthetic data"""
    
    print("="*60)
    print("LSTM TIME SERIES TRADING PREDICTOR - TRAINING")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = ModelConfig.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")
    
    # Generate synthetic OHLCV data
    print("Generating synthetic OHLCV data...")
    df = DataLoader.generate_synthetic_ohlcv_data(num_samples=2000)
    print(f"Generated data shape: {df.shape}")
    
    # Preprocess
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor(scaler_type="minmax")
    X, y = preprocessor.process_ohlcv_data(df, seq_length=DataConfig.SEQ_LENGTH)
    print(f"Sequences shape: X={X.shape}, y={y.shape}")
    
    # Train/Val/Test split
    print("\nSplitting data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        DataLoader.train_val_test_split(
            X, y,
            train_ratio=ModelConfig.TRAIN_SPLIT,
            val_ratio=ModelConfig.VAL_SPLIT
        )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_time_series_loaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=ModelConfig.BATCH_SIZE
    )
    print(f"Batch size: {ModelConfig.BATCH_SIZE}")
    
    # Create model
    print("\nCreating LSTM model...")
    model = LSTMPredictor(
        input_size=ModelConfig.LSTM_INPUT_SIZE,
        hidden_dim=ModelConfig.LSTM_HIDDEN_DIM,
        num_layers=ModelConfig.LSTM_NUM_LAYERS,
        dropout=ModelConfig.LSTM_DROPOUT
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = LSTMTrainer(model, device=device, learning_rate=ModelConfig.LEARNING_RATE)
    
    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING...")
    print("="*60 + "\n")
    
    history = trainer.train(
        train_loader, val_loader,
        epochs=ModelConfig.EPOCHS,
        checkpoint_dir="checkpoints"
    )
    
    # Test
    print("\n" + "="*60)
    print("TESTING MODEL...")
    print("="*60 + "\n")
    
    test_loss = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_checkpoint("checkpoints/lstm_final.pt", ModelConfig.EPOCHS, test_loss)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return history, test_loss


if __name__ == "__main__":
    history, test_loss = train_lstm_model()
    
    print("\nTraining completed successfully!")
    print(f"Final test loss: {test_loss:.4f}")
    print("\nCheckpoints saved in 'checkpoints/' directory")
    print("Use 'inference_lstm.py' to make predictions with the trained model")
