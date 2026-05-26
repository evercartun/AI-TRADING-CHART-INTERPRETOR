"""
Main demo script showcasing all AI trading prediction functionality
Usage: python demo.py
"""

import sys
from pathlib import Path


def print_banner():
    """Print project banner"""
    print("\n" + "="*70)
    print(" "*15 + "🔮 AI TRADING CHART PREDICTOR 🔮")
    print(" "*10 + "Powered by PyTorch, TensorFlow & Scikit-Learn")
    print("="*70 + "\n")


def print_menu():
    """Print main menu"""
    print("SELECT AN OPTION:\n")
    print("1. Train LSTM Time Series Model")
    print("2. Train CNN Image Classifier")
    print("3. Make LSTM Predictions")
    print("4. Make CNN Predictions")
    print("5. Ensemble Predictions (Combined)")
    print("6. View Project Information")
    print("7. Exit\n")


def option_train_lstm():
    """Train LSTM model"""
    print("\n" + "="*70)
    print("TRAINING LSTM TIME SERIES MODEL...")
    print("="*70 + "\n")
    print("This will generate synthetic OHLCV data and train an LSTM model")
    print("for time series price prediction.\n")
    
    import train_lstm
    train_lstm.train_lstm_model()


def option_train_cnn():
    """Train CNN model"""
    print("\n" + "="*70)
    print("TRAINING CNN IMAGE CLASSIFIER...")
    print("="*70 + "\n")
    print("This will generate synthetic chart images and train a CNN model")
    print("for trend classification.\n")
    
    import train_cnn
    train_cnn.train_cnn_model()


def option_lstm_inference():
    """LSTM inference"""
    print("\n" + "="*70)
    print("LSTM MODEL - INFERENCE")
    print("="*70 + "\n")
    
    import inference_lstm
    inference_lstm.demo_lstm_prediction()
    inference_lstm.predict_custom_data()


def option_cnn_inference():
    """CNN inference"""
    print("\n" + "="*70)
    print("CNN MODEL - INFERENCE")
    print("="*70 + "\n")
    
    import inference_cnn
    inference_cnn.demo_cnn_prediction()


def option_ensemble_inference():
    """Ensemble inference"""
    print("\n" + "="*70)
    print("ENSEMBLE MODEL - COMBINED INFERENCE")
    print("="*70 + "\n")
    
    import inference_ensemble
    inference_ensemble.print_model_info()
    inference_ensemble.demo_ensemble_prediction()
    inference_ensemble.predict_with_real_data_example()


def option_project_info():
    """Print project information"""
    print("\n" + "="*70)
    print("PROJECT INFORMATION")
    print("="*70 + "\n")
    
    print("""
📊 PROJECT STRUCTURE:
├── config.py              - Configuration and hyperparameters
├── models/
│   ├── pytorch_models.py  - LSTM and CNN models using PyTorch
│   └── tensorflow_models.py - TensorFlow/Keras implementations
├── utils/
│   ├── preprocessing.py   - Data preprocessing and normalization
│   ├── torch_data.py      - PyTorch Dataset classes
│   ├── trainer.py         - Training utilities and trainer classes
│   └── inference.py       - Inference engines and predictors
├── train_lstm.py          - Training script for LSTM
├── train_cnn.py           - Training script for CNN
├── inference_lstm.py      - LSTM inference demo
├── inference_cnn.py       - CNN inference demo
├── inference_ensemble.py  - Ensemble inference
└── data/                  - Data storage directory

🧠 MODELS INCLUDED:

1. LSTM TIME SERIES MODEL
   - Predicts price movements from historical OHLCV data
   - Input: 60-step time series with 5 features (Open, High, Low, Close, Volume)
   - Output: Predicted price change
   - Architecture: Multi-layer LSTM + Attention + Fully Connected Layers
   - Framework: PyTorch

2. CNN IMAGE CLASSIFIER
   - Classifies trading chart patterns and trends
   - Input: 224×224 RGB chart images
   - Output: Trend classification (Downtrend, Neutral, Uptrend)
   - Architecture: Convolutional layers + Max Pooling + FC layers
   - Framework: PyTorch

3. ENSEMBLE PREDICTOR
   - Combines LSTM and CNN predictions
   - Provides robust trading signals
   - Calculates confidence scores
   - Handles multi-modal input (time series + images)

🚀 QUICK START:

1. Install dependencies:
   pip install -r requirements.txt

2. Train models:
   python train_lstm.py
   python train_cnn.py

3. Make predictions:
   python inference_lstm.py
   python inference_cnn.py <image_path>
   python inference_ensemble.py

4. Use in code:
   from utils.inference import EnsemblePredictor
   ensemble = EnsemblePredictor('checkpoints/lstm_final.pt', 'checkpoints/cnn_final.pt')
   result = ensemble.predict_combined(ohlcv_data, image_path='chart.png')

📈 FEATURES:

✓ No API keys required - models built from scratch
✓ Support for PyTorch and TensorFlow
✓ LSTM for time series prediction
✓ CNN for image analysis
✓ Ensemble learning for robust predictions
✓ Data preprocessing and normalization
✓ Training with early stopping and checkpointing
✓ Confidence estimation
✓ Synthetic data generation for testing
✓ Easy-to-use inference interfaces

🔧 CUSTOMIZATION:

- Modify config.py to adjust hyperparameters
- Change model architectures in models/
- Add real trading data in CSV format
- Train custom models with your data
- Export trained models for deployment

📚 SCIKIT-LEARN INTEGRATION:

Preprocessing utilities use scikit-learn:
- MinMaxScaler / StandardScaler for normalization
- Train/test split functionality
- Future: Add ML algorithms (Random Forest, SVM, etc.)

    """)
    
    print("="*70 + "\n")


def main():
    """Main application loop"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == '1':
                option_train_lstm()
            elif choice == '2':
                option_train_cnn()
            elif choice == '3':
                option_lstm_inference()
            elif choice == '4':
                option_cnn_inference()
            elif choice == '5':
                option_ensemble_inference()
            elif choice == '6':
                option_project_info()
            elif choice == '7':
                print("\nGoodbye! 👋\n")
                break
            else:
                print("Invalid choice. Please try again.\n")
        
        except KeyboardInterrupt:
            print("\n\nProgram interrupted. Goodbye! 👋\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
