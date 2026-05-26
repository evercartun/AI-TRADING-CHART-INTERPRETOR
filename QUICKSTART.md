# Quick Start Guide - AI Trading Chart Predictor

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train Models (First Time)
```bash
# Train LSTM on synthetic data (~2-3 minutes)
python train_lstm.py

# Train CNN on synthetic data (~2-3 minutes)  
python train_cnn.py
```

### Step 3: Make Predictions
```bash
# Option A: Interactive Demo
python demo.py

# Option B: LSTM predictions
python inference_lstm.py

# Option C: CNN predictions on chart image
python inference_cnn.py path/to/chart.png

# Option D: Ensemble (both models)
python inference_ensemble.py
```

---

## Common Use Cases

### Use Case 1: Predict Next Price Movement
```python
from utils.inference import LSTMInferencer
from utils.preprocessing import DataLoader
import pandas as pd

# Load your OHLCV data
df = pd.read_csv('your_data.csv')
sequence = df[['open', 'high', 'low', 'close', 'volume']].iloc[-60:].values

# Predict
predictor = LSTMInferencer('checkpoints/lstm_final.pt')
price_change, confidence = predictor.predict_with_confidence(sequence)

print(f"Predicted change: ${price_change:.4f}")
print(f"Confidence: {confidence:.2%}")
```

### Use Case 2: Classify Chart Trend
```python
from utils.inference import CNNInferencer

# Analyze chart image
classifier = CNNInferencer('checkpoints/cnn_final.pt')
trend, confidence = classifier.predict('my_chart.png')

print(f"Trend: {trend}")  # 'Uptrend', 'Downtrend', or 'Neutral'
print(f"Confidence: {confidence:.2%}")
```

### Use Case 3: Combined Analysis
```python
from utils.inference import EnsemblePredictor

ensemble = EnsemblePredictor(
    'checkpoints/lstm_final.pt',
    'checkpoints/cnn_final.pt'
)

result = ensemble.predict_combined(ohlcv_sequence, 'chart.png')

print(f"Signal: {result['ensemble_signal']}")           # 'up', 'down', 'neutral'
print(f"Confidence: {result['combined_confidence']:.2%}")
```

---

## Using Your Own Data

### CSV Format (OHLCV)
```csv
date,open,high,low,close,volume
2024-01-01,100.0,101.5,99.0,100.5,1000000
2024-01-02,100.5,102.0,99.5,101.0,1100000
...
```

### Training on Your Data
```python
import pandas as pd
from utils.preprocessing import DataPreprocessor
from models.pytorch_models import LSTMPredictor
from utils.trainer import LSTMTrainer

# Load data
df = pd.read_csv('my_trading_data.csv')

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.process_ohlcv_data(df)

# Split and create loaders
from utils.torch_data import create_time_series_loaders

# Train
model = LSTMPredictor()
trainer = LSTMTrainer(model)
trainer.train(train_loader, val_loader, epochs=50)
trainer.save_checkpoint('my_model.pt', 0, 0)
```

---

## Key Files & What They Do

| File | Purpose |
|------|---------|
| `config.py` | Hyperparameters and settings |
| `models/pytorch_models.py` | LSTM and CNN architectures |
| `utils/preprocessing.py` | Data cleaning and normalization |
| `utils/trainer.py` | Model training loops |
| `utils/inference.py` | Make predictions |
| `train_lstm.py` | Train LSTM model |
| `train_cnn.py` | Train CNN model |
| `inference_lstm.py` | Demo LSTM predictions |
| `inference_cnn.py` | Demo CNN predictions |
| `inference_ensemble.py` | Demo ensemble predictions |

---

## Troubleshooting

### Issue: "No module named torch"
**Solution**: Run `pip install -r requirements.txt`

### Issue: CUDA out of memory
**Solution**: Change `DEVICE = "cpu"` in `config.py`

### Issue: Model checkpoint not found
**Solution**: Train the model first with `python train_lstm.py`

### Issue: Image not found for CNN
**Solution**: Provide full path or check file exists

### Issue: Slow predictions
**Solution**: Use `device="cuda"` if GPU available

---

## Performance Tips

1. **Use GPU**: Set `DEVICE = "cuda"` if NVIDIA GPU available
2. **Batch Processing**: Use `predict_batch()` for multiple samples
3. **Confidence Filtering**: Only act on predictions > 0.7 confidence
4. **Cache Preprocessor**: Reuse preprocessor for same data scale

---

## Model Architecture Summary

**LSTM Model:**
- 3-layer LSTM (128 hidden units)
- Attention mechanism
- Input: 60 timesteps × 5 features (OHLCV)
- Output: Single value (price prediction)

**CNN Model:**
- 4 convolutional blocks with batch norm
- Global average pooling
- Input: 224×224 RGB image
- Output: 3 classes (Up/Neutral/Down)

**Ensemble:**
- LSTM for time series analysis
- CNN for visual pattern recognition
- Combined confidence scoring

---

## Next Steps

1. ✅ Run `python demo.py` for interactive walkthrough
2. ✅ Train models on your real trading data
3. ✅ Fine-tune hyperparameters in `config.py`
4. ✅ Deploy as REST API or integrate with trading bot
5. ✅ Extend with additional indicators

---

## Command Reference

```bash
# Training
python train_lstm.py              # Train LSTM model
python train_cnn.py               # Train CNN model

# Inference
python inference_lstm.py           # Test LSTM
python inference_cnn.py <img>     # Test CNN on image
python inference_ensemble.py      # Test both models

# Demo
python demo.py                     # Interactive demo

# Using models in Python
from models.pytorch_models import LSTMPredictor, CNNChartClassifier
from utils.inference import EnsemblePredictor
```

---

Happy trading! 📈

Questions? Check the README.md for detailed documentation.
