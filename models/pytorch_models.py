"""
Neural network models for trading prediction using PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class LSTMPredictor(nn.Module):
    """
    LSTM-based model for time series price prediction
    """
    
    def __init__(self, 
                 input_size: int = ModelConfig.LSTM_INPUT_SIZE,
                 hidden_dim: int = ModelConfig.LSTM_HIDDEN_DIM,
                 num_layers: int = ModelConfig.LSTM_NUM_LAYERS,
                 dropout: float = ModelConfig.LSTM_DROPOUT,
                 output_size: int = ModelConfig.LSTM_OUTPUT_SIZE):
        """
        Initialize LSTM predictor
        
        Args:
            input_size: Number of features (OHLCV = 5)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Output dimension (1 for price prediction)
        """
        super(LSTMPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention layer (optional but effective)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=dropout
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Prediction tensor of shape (batch_size, output_size)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output and apply attention
        last_out = attn_out[:, -1, :]
        
        # Fully connected layers with dropout and ReLU
        x = self.relu(self.fc1(last_out))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class CNNChartClassifier(nn.Module):
    """
    CNN-based model for chart image analysis and trend classification
    Classifies charts as: 0=Downtrend, 1=Neutral, 2=Uptrend
    """
    
    def __init__(self,
                 num_channels: int = ModelConfig.CNN_NUM_CHANNELS,
                 num_classes: int = ModelConfig.CNN_NUM_CLASSES,
                 dropout: float = ModelConfig.CNN_DROPOUT):
        """
        Initialize CNN classifier
        
        Args:
            num_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (3 for trend classification)
            dropout: Dropout rate
        """
        super(CNNChartClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional blocks
        # Block 1: 3 -> 64 channels
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 64 -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: 128 -> 256 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4: 256 -> 512 channels
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class CNNImageFeatureExtractor(nn.Module):
    """
    Lightweight CNN for feature extraction from chart images
    Can be used as backbone for transfer learning or ensemble
    """
    
    def __init__(self, num_channels: int = 3, feature_dim: int = 256):
        """
        Initialize feature extractor
        
        Args:
            num_channels: Number of input channels
            feature_dim: Dimension of extracted features
        """
        super(CNNImageFeatureExtractor, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(num_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection
        self.fc = nn.Linear(128, feature_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from image
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class HybridTradingPredictor(nn.Module):
    """
    Hybrid model combining LSTM (time series) and CNN (image) features
    for more robust trading predictions
    """
    
    def __init__(self,
                 lstm_hidden_dim: int = ModelConfig.LSTM_HIDDEN_DIM,
                 num_lstm_layers: int = ModelConfig.LSTM_NUM_LAYERS,
                 lstm_input_size: int = ModelConfig.LSTM_INPUT_SIZE,
                 cnn_feature_dim: int = 128,
                 dropout: float = 0.3):
        """
        Initialize hybrid model
        """
        super(HybridTradingPredictor, self).__init__()
        
        # LSTM component
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # CNN feature extractor
        self.cnn_feature_extractor = CNNImageFeatureExtractor(
            num_channels=3,
            feature_dim=cnn_feature_dim
        )
        
        # Fusion layers
        combined_dim = lstm_hidden_dim + cnn_feature_dim
        self.fusion_fc1 = nn.Linear(combined_dim, combined_dim // 2)
        self.fusion_fc2 = nn.Linear(combined_dim // 2, 64)
        self.output_fc = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, lstm_input: torch.Tensor, 
                image_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with both LSTM and image inputs
        
        Args:
            lstm_input: Time series input of shape (batch, seq_len, features)
            image_input: Image input of shape (batch, 3, height, width)
            
        Returns:
            Prediction of shape (batch, 1)
        """
        # LSTM processing
        lstm_out, _ = self.lstm(lstm_input)
        lstm_features = lstm_out[:, -1, :]
        
        # CNN processing
        cnn_features = self.cnn_feature_extractor(image_input)
        
        # Fusion
        combined = torch.cat([lstm_features, cnn_features], dim=1)
        x = self.relu(self.fusion_fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fusion_fc2(x))
        x = self.dropout(x)
        x = self.output_fc(x)
        
        return x
