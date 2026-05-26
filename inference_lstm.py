"""
Inference script for LSTM model - predict trading prices from time series data
Usage: python inference_lstm.py
"""

import numpy as np
from pathlib import Path

from utils.preprocessing import DataPreprocessor, DataLoader
from utils.inference import LSTMInferencer, create_prediction_summary


def demo_lstm_prediction():
    """Demonstrate LSTM inference"""
    
    print("="*60)
    print("LSTM TIME SERIES - INFERENCE DEMO")
    print("="*60 + "\n")
    
    # Check if model exists
    model_path = Path("checkpoints/lstm_final.pt")
    if not model_path.exists():
        print("ERROR: Model not found at checkpoints/lstm_final.pt")
        print("Please run 'python train_lstm.py' first to train the model.\n")
        
        # Create demo data anyway
        print("Creating demo with synthetic data...\n")
        df = DataLoader.generate_synthetic_ohlcv_data(num_samples=100)
        data = df[['open', 'high', 'low', 'close', 'volume']].values[-60:]  # Last 60 days
        
        print("Demo OHLCV Data (last 10 rows):")
        print(data[-10:])
        print("\nTo get actual predictions, train the model with 'python train_lstm.py'")
        return
    
    print("Loading model...\n")
    inferencer = LSTMInferencer(str(model_path), device="cpu")
    
    # Generate test data
    print("Generating test data...")
    df = DataLoader.generate_synthetic_ohlcv_data(num_samples=100)
    test_data = df[['open', 'high', 'low', 'close', 'volume']].values
    
    # Make predictions on multiple sequences
    print("\nMaking predictions...\n")
    
    for i in range(5):
        sequence = test_data[i:i+60]  # 60-day window
        
        if len(sequence) == 60:
            prediction, confidence = inferencer.predict_with_confidence(sequence)
            
            print(f"Sequence {i+1}:")
            print(f"  Last Close Price: ${sequence[-1, 3]:.2f}")
            print(f"  Predicted Change: ${prediction:.4f}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Signal: {'UP' if prediction > 0 else 'DOWN'}\n")


def predict_custom_data():
    """Example: Make prediction on custom OHLCV data"""
    
    print("\n" + "="*60)
    print("CUSTOM DATA PREDICTION EXAMPLE")
    print("="*60 + "\n")
    
    model_path = Path("checkpoints/lstm_final.pt")
    if not model_path.exists():
        print("Model not found. Train first with 'python train_lstm.py'\n")
        return
    
    inferencer = LSTMInferencer(str(model_path), device="cpu")
    
    # Create custom OHLCV data (60 days)
    print("Example: Creating custom 60-day OHLCV sequence...")
    custom_data = np.array([
        [100.0, 101.5, 99.0, 100.5, 1000000],  # Day 1
        [100.5, 102.0, 99.5, 101.0, 1100000],  # Day 2
        # ... add more days up to 60
    ] + [[100.0 + i*0.5, 101.5 + i*0.5, 99.0 + i*0.5, 100.5 + i*0.5, 1000000] 
         for i in range(57)])  # Fill rest with pattern
    
    print(f"Data shape: {custom_data.shape}")
    print(f"First row: {custom_data[0]}")
    print(f"Last row: {custom_data[-1]}\n")
    
    # Make prediction
    prediction, confidence = inferencer.predict_with_confidence(custom_data)
    
    result = {
        'lstm_prediction': prediction,
        'lstm_confidence': confidence
    }
    
    summary = create_prediction_summary(result)
    print(summary)


if __name__ == "__main__":
    demo_lstm_prediction()
    predict_custom_data()
