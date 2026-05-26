# AI Trading Chart Predictor

A comprehensive AI system for predicting trading charts using deep learning models built from scratch with **PyTorch**, **TensorFlow**, and **Scikit-Learn**. No API keys required!

## 🌟 Features

- **LSTM Time Series Model**: Predicts price movements from historical OHLCV data
- **CNN Image Classifier**: Analyzes trading chart images to classify trends
- **Ensemble Learning**: Combines multiple models for robust predictions
- **Confidence Scoring**: Get confidence estimates with predictions
- **No External APIs**: All models built from scratch using open-source libraries
- **Synthetic Data Generation**: Generate test data automatically
- **Easy-to-Use Interfaces**: Simple APIs for training and inference

## 🏗️ Project Structure

```
├── config.py              # Configuration and hyperparameters
├── models/
│   ├── pytorch_models.py  # LSTM and CNN using PyTorch
│   └── tensorflow_models.py # Alternative TensorFlow implementations
├── utils/
│   ├── preprocessing.py   # Data preprocessing and normalization
│   ├── torch_data.py      # PyTorch Dataset classes
│   ├── trainer.py         # Training utilities
│   └── inference.py       # Inference engines
├── train_lstm.py          # LSTM training script
├── train_cnn.py           # CNN training script
├── inference_*.py         # Inference scripts
├── demo.py                # Interactive demo
├── checkpoints/           # Saved model checkpoints
└── data/                  # Data storage
```

## 🧠 Models

### 1. LSTM Time Series Model
- **Input**: 60-step sequences with OHLCV data (5 features)
- **Output**: Price prediction (continuous value)
- **Architecture**: Multi-layer LSTM + Attention + Fully Connected
- **Framework**: PyTorch
- **Use Case**: Predict next price movement from historical data

### 2. CNN Image Classifier
- **Input**: 224×224 RGB chart images
- **Output**: Trend classification (Downtrend, Neutral, Uptrend)
- **Architecture**: Convolutional layers + BatchNorm + Pooling
- **Framework**: PyTorch
- **Use Case**: Classify trading patterns from chart images

### 3. Hybrid Ensemble
- Combines LSTM (time series) and CNN (images)
- Provides robust trading signals
- Calculates combined confidence scores

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- `torch` - PyTorch for deep learning
- `tensorflow` - TensorFlow/Keras (alternative)
- `scikit-learn` - Machine learning utilities
- `numpy`, `pandas` - Data manipulation
- `opencv-python` - Image processing
- `matplotlib` - Visualization

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models (on synthetic data)
```bash
# Train LSTM model
python train_lstm.py

# Train CNN model
python train_cnn.py
```

### 3. Make predictions
```bash
# LSTM predictions on time series
python inference_lstm.py

# CNN predictions on chart images
python inference_cnn.py <path_to_image>

# Ensemble predictions (both models)
python inference_ensemble.py
```

### 4. Interactive demo
```bash
python demo.py
```

## 💻 Usage Examples

### LSTM Prediction
```python
from utils.inference import LSTMInferencer
from utils.preprocessing import DataLoader
import numpy as np

# Load or prepare OHLCV data (60 steps × 5 features)
df = DataLoader.generate_synthetic_ohlcv_data(100)
sequence = df[['open', 'high', 'low', 'close', 'volume']].values[-60:]

# Make prediction
inferencer = LSTMInferencer('checkpoints/lstm_final.pt')
prediction, confidence = inferencer.predict_with_confidence(sequence)

print(f"Prediction: {prediction:.4f}")
print(f"Confidence: {confidence:.2%}")
print(f"Signal: {'UP' if prediction > 0 else 'DOWN'}")
```

### CNN Prediction
```python
from utils.inference import CNNInferencer

# Predict on chart image
inferencer = CNNInferencer('checkpoints/cnn_final.pt')
trend, confidence = inferencer.predict('path/to/chart.png')

print(f"Trend: {trend}")
print(f"Confidence: {confidence:.2%}")
```

### Ensemble Prediction
```python
from utils.inference import EnsemblePredictor

# Combine LSTM and CNN
ensemble = EnsemblePredictor(
    lstm_model_path='checkpoints/lstm_final.pt',
    cnn_model_path='checkpoints/cnn_final.pt',
    device='cuda'
)

# Make prediction with both models
result = ensemble.predict_combined(
    ohlcv_data=sequence,
    image_path='chart.png'
)

print(result['ensemble_signal'])  # 'up', 'down', or 'conflicting'
print(result['combined_confidence'])
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model hyperparameters
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 3
LSTM_DROPOUT = 0.2

CNN_IMG_HEIGHT = 224
CNN_IMG_WIDTH = 224
CNN_DROPOUT = 0.3

# Training settings
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Early stopping patience
EARLY_STOPPING_PATIENCE = 15
```

## 📊 Training with Real Data

To train on your own trading data:

```python
import pandas as pd
from utils.preprocessing import DataPreprocessor
from utils.torch_data import create_time_series_loaders

# Load your CSV with OHLCV data
df = pd.read_csv('your_trading_data.csv')

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.process_ohlcv_data(df, seq_length=60)

# Create data loaders
train_loader, val_loader, test_loader = create_time_series_loaders(
    X_train, y_train, X_val, y_val, X_test, y_test
)

# Train model
from utils.trainer import LSTMTrainer
trainer = LSTMTrainer(model)
trainer.train(train_loader, val_loader, epochs=100)
```

## 🎯 Prediction Confidence

Both models provide confidence scores:

- **LSTM**: Estimated by multiple perturbation samples
- **CNN**: From softmax probability distribution
- **Ensemble**: Average of individual confidences

Filter predictions by confidence threshold in `config.py`:

```python
CONFIDENCE_THRESHOLD = 0.7  # Only act on predictions above this
```

## 🧪 Testing & Validation

- Automatic train/val/test split (70/15/15)
- Early stopping to prevent overfitting
- Checkpoint saving every N epochs
- Test metrics displayed after training

## 📈 Supported File Formats

- **Time Series Data**: CSV with columns: open, high, low, close, volume
- **Chart Images**: PNG, JPG, JPEG, BMP
- **Models**: PyTorch (.pt) and TensorFlow (.h5)

## 🔐 Security

- ✅ No API keys required
- ✅ No external service calls
- ✅ All processing local to your machine
- ✅ Models can be run offline
- ✅ Full source code included

## 📝 Model Details

### Data Preprocessing
- MinMax or Standard normalization via scikit-learn
- Automatic OHLCV sequence creation
- Image resizing and RGB normalization
- Data augmentation for images

### Training Features
- Adam optimizer with learning rate scheduling
- Gradient clipping to prevent exploding gradients
- Batch normalization in CNN
- Dropout regularization
- Early stopping with patience

### Inference Features
- Batch prediction support
- Uncertainty estimation
- Multi-modal input (time series + images)
- Ensemble voting
- Class probability distributions

## 🤝 Contributing

Feel free to extend the project:
- Add new model architectures
- Implement additional ML algorithms (Random Forest, SVM)
- Add technical indicators
- Support for real-time predictions
- Integration with trading platforms

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙋 Support

For issues or questions:
1. Check the demo script for usage examples
2. Review configuration in `config.py`
3. Examine the training/inference scripts
4. Check model outputs in checkpoints/

## 🎓 Learning Resources

The codebase demonstrates:
- PyTorch model architecture design
- Time series deep learning with LSTM
- Computer vision with CNN
- Ensemble learning techniques
- Data preprocessing and normalization
- Model training and evaluation
- Inference optimization

Happy trading! 📈💰
