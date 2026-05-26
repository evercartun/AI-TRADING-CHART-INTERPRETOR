"""
Ensemble inference - Combine LSTM and CNN predictions for robust trading signals
Usage: python inference_ensemble.py
"""

import numpy as np
from pathlib import Path

from utils.preprocessing import DataPreprocessor, DataLoader
from utils.inference import EnsemblePredictor, create_prediction_summary


def demo_ensemble_prediction():
    """Demonstrate ensemble inference combining both models"""
    
    print("="*60)
    print("ENSEMBLE TRADING PREDICTOR - COMBINED INFERENCE")
    print("="*60 + "\n")
    
    lstm_model_path = Path("checkpoints/lstm_final.pt")
    cnn_model_path = Path("checkpoints/cnn_final.pt")
    
    if not lstm_model_path.exists() or not cnn_model_path.exists():
        print("WARNING: One or both models not found!")
        print(f"  LSTM model: {'✓ Found' if lstm_model_path.exists() else '✗ Not found'}")
        print(f"  CNN model: {'✓ Found' if cnn_model_path.exists() else '✗ Not found'}\n")
        print("To train models, run:")
        print("  python train_lstm.py")
        print("  python train_cnn.py\n")
        return
    
    print("Loading models...\n")
    ensemble = EnsemblePredictor(
        lstm_model_path=str(lstm_model_path) if lstm_model_path.exists() else None,
        cnn_model_path=str(cnn_model_path) if cnn_model_path.exists() else None,
        device="cpu"
    )
    
    # Generate test data
    print("Generating test OHLCV data...\n")
    df = DataLoader.generate_synthetic_ohlcv_data(num_samples=100)
    test_data = df[['open', 'high', 'low', 'close', 'volume']].values
    
    # Get test chart image if available
    test_images_dir = Path("data/synthetic_charts")
    test_image = None
    if test_images_dir.exists():
        images = list(test_images_dir.glob("*.png"))
        if images:
            test_image = str(images[0])
    
    # Make ensemble predictions
    print("Making ensemble predictions...\n")
    
    for i in range(3):
        sequence = test_data[i:i+60]
        
        if len(sequence) == 60:
            print(f"Prediction {i+1}:")
            print("-" * 40)
            
            result = ensemble.predict_combined(sequence, image_path=test_image)
            
            if 'lstm_prediction' in result:
                print(f"LSTM Prediction: {result['lstm_prediction']:.4f}")
                print(f"LSTM Confidence: {result['lstm_confidence']:.2%}")
            
            if 'cnn_prediction' in result:
                print(f"CNN Prediction: {result['cnn_prediction']}")
                print(f"CNN Confidence: {result['cnn_confidence']:.2%}")
            
            if 'ensemble_signal' in result:
                print(f"\nEnsemble Signal: {result['ensemble_signal'].upper()}")
                print(f"Combined Confidence: {result['combined_confidence']:.2%}")
            
            print()


def predict_with_real_data_example():
    """Example showing how to use the ensemble with real trading data"""
    
    print("\n" + "="*60)
    print("ENSEMBLE WITH REAL DATA - EXAMPLE")
    print("="*60 + "\n")
    
    lstm_model_path = Path("checkpoints/lstm_final.pt")
    cnn_model_path = Path("checkpoints/cnn_final.pt")
    
    if not lstm_model_path.exists() or not cnn_model_path.exists():
        print("Models not found. Train first with train_lstm.py and train_cnn.py\n")
        return
    
    print("Example usage with real trading data:\n")
    print("""
# Load your real OHLCV data
import pandas as pd
ohlcv_data = pd.read_csv('your_trading_data.csv')
sequence = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].iloc[-60:].values

# Load chart image if available
chart_image_path = 'path/to/chart.png'

# Create ensemble predictor
from utils.inference import EnsemblePredictor
ensemble = EnsemblePredictor(
    lstm_model_path='checkpoints/lstm_final.pt',
    cnn_model_path='checkpoints/cnn_final.pt',
    device='cuda'  # or 'cpu'
)

# Make prediction
result = ensemble.predict_combined(sequence, image_path=chart_image_path)

# View results
from utils.inference import create_prediction_summary
print(create_prediction_summary(result))
    """)


def print_model_info():
    """Print information about available models"""
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60 + "\n")
    
    from models.pytorch_models import LSTMPredictor, CNNChartClassifier
    
    print("LSTM Time Series Model:")
    print("  - Input: Time series sequences (60 steps, 5 features: OHLCV)")
    print("  - Output: Price prediction (continuous value)")
    print("  - Architecture: LSTM with attention + FC layers")
    print("  - Use case: Predict future price movements from historical data")
    
    print("\nCNN Image Classifier:")
    print("  - Input: Chart images (224x224x3 RGB)")
    print("  - Output: Trend classification (3 classes: Down, Neutral, Up)")
    print("  - Architecture: ConvNet with pooling + FC layers")
    print("  - Use case: Classify trading chart trends from images")
    
    print("\nEnsemble Strategy:")
    print("  - Combines LSTM and CNN predictions")
    print("  - LSTM Signal: 'up' if prediction > 0, else 'down'")
    print("  - CNN Signal: Direct class prediction (Up/Neutral/Down)")
    print("  - Combined Signal: Agreement or 'conflicting' if disagreement")
    print("  - Confidence: Average of individual confidences")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print_model_info()
    demo_ensemble_prediction()
    predict_with_real_data_example()
