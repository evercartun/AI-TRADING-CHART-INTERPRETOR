"""
Configuration settings for the Trading Chart Prediction AI
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
UTILS_DIR = PROJECT_ROOT / "utils"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Configuration
class ModelConfig:
    # LSTM Model
    LSTM_HIDDEN_DIM = 128
    LSTM_NUM_LAYERS = 3
    LSTM_DROPOUT = 0.2
    LSTM_INPUT_SIZE = 5  # OHLCV
    LSTM_OUTPUT_SIZE = 1  # Prediction
    LSTM_SEQ_LENGTH = 60  # 60 time steps
    
    # CNN Image Model
    CNN_NUM_CHANNELS = 3  # RGB
    CNN_IMG_HEIGHT = 224
    CNN_IMG_WIDTH = 224
    CNN_NUM_CLASSES = 3  # Up, Down, Neutral
    CNN_DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    DEVICE = "cuda"  # or "cpu"
    
    # Data
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    NORMALIZE = True
    
    # Checkpointing
    SAVE_FREQ = 5  # Save every N epochs
    EARLY_STOPPING_PATIENCE = 15

# Data Configuration
class DataConfig:
    # Time series parameters
    SEQ_LENGTH = 60
    FORECAST_HORIZON = 1  # Predict 1 step ahead
    OHLCV_COLS = ["open", "high", "low", "close", "volume"]
    
    # Image parameters
    IMG_SIZE = (224, 224)
    IMG_CHANNELS = 3
    
    # Preprocessing
    SCALER_TYPE = "minmax"  # or "standard"

# Training Configuration
class TrainingConfig:
    DEVICE = "cuda"
    NUM_WORKERS = 4
    SEED = 42
    LOG_INTERVAL = 10
    
# Inference Configuration
class InferenceConfig:
    CONFIDENCE_THRESHOLD = 0.7
    ENSEMBLE = True  # Use ensemble of models
