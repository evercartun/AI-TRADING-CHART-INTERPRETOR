"""
Neural network models for trading prediction using TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from config import ModelConfig


class LSTMPredictorTF(Model):
    """
    TensorFlow/Keras LSTM model for time series price prediction
    """
    
    def __init__(self,
                 input_size: int = ModelConfig.LSTM_INPUT_SIZE,
                 hidden_dim: int = ModelConfig.LSTM_HIDDEN_DIM,
                 num_layers: int = ModelConfig.LSTM_NUM_LAYERS,
                 dropout: float = ModelConfig.LSTM_DROPOUT):
        """
        Initialize TensorFlow LSTM model
        
        Args:
            input_size: Number of features (OHLCV = 5)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMPredictorTF, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # LSTM layers
        self.lstm_layers = [
            layers.LSTM(hidden_dim, return_sequences=(i < num_layers - 1),
                       dropout=dropout, input_shape=(None, input_size))
            for i in range(num_layers)
        ]
        
        # Attention layer
        self.attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=hidden_dim // 4,
            dropout=dropout
        )
        
        # Fully connected layers
        self.fc_layers = [
            layers.Dense(hidden_dim // 2, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(hidden_dim // 4, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(1)
        ]
    
    def call(self, x, training=False):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            training: Whether in training mode
            
        Returns:
            Prediction tensor of shape (batch_size, 1)
        """
        # LSTM processing
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i == 0:
                x = lstm_layer(x, training=training)
            else:
                x = lstm_layer(x, training=training)
        
        # Add batch dimension back if needed
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)
        
        # Apply attention
        x = self.attention(x, x, training=training)
        
        # Take last output
        x = x[:, -1, :]
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x, training=training)
        
        return x
    
    def get_config(self):
        return {
            "input_size": self.input_size,
            "hidden_dim": self.hidden_dim
        }


class CNNChartClassifierTF(Model):
    """
    TensorFlow/Keras CNN model for chart image analysis and trend classification
    """
    
    def __init__(self,
                 num_channels: int = ModelConfig.CNN_NUM_CHANNELS,
                 num_classes: int = ModelConfig.CNN_NUM_CLASSES,
                 dropout: float = ModelConfig.CNN_DROPOUT):
        """
        Initialize CNN classifier
        
        Args:
            num_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (3 for trend)
            dropout: Dropout rate
        """
        super(CNNChartClassifierTF, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional base
        self.conv_base = Sequential([
            layers.Conv2D(64, 3, activation='relu', padding='same',
                         input_shape=(224, 224, num_channels)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            layers.Conv2D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            layers.Conv2D(512, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
        ])
        
        # Classification head
        self.classifier = Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(num_classes)
        ])
    
    def call(self, x, training=False):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 224, 224, 3)
            training: Whether in training mode
            
        Returns:
            Class logits of shape (batch, num_classes)
        """
        x = self.conv_base(x, training=training)
        x = self.classifier(x, training=training)
        return x


def create_lstm_model_tf(seq_length: int = ModelConfig.LSTM_SEQ_LENGTH,
                         input_size: int = ModelConfig.LSTM_INPUT_SIZE) -> Model:
    """
    Create a TensorFlow Sequential LSTM model for time series prediction
    
    Args:
        seq_length: Length of input sequences
        input_size: Number of features
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        layers.LSTM(128, return_sequences=True, 
                   input_shape=(seq_length, input_size)),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_cnn_model_tf(img_height: int = ModelConfig.CNN_IMG_HEIGHT,
                        img_width: int = ModelConfig.CNN_IMG_WIDTH,
                        num_classes: int = ModelConfig.CNN_NUM_CLASSES) -> Model:
    """
    Create a TensorFlow Sequential CNN model for image classification
    
    Args:
        img_height: Image height
        img_width: Image width
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model


def create_transfer_learning_model_tf(num_classes: int = ModelConfig.CNN_NUM_CLASSES) -> Model:
    """
    Create a transfer learning model using pre-trained MobileNetV2
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom classification head
    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model
