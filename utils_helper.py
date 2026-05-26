"""
Utility script for common operations and project management
"""

import os
import shutil
from pathlib import Path
import torch
import numpy as np


def print_project_summary():
    """Print comprehensive project summary"""
    print("\n" + "="*70)
    print("AI TRADING CHART PREDICTOR - PROJECT SUMMARY")
    print("="*70 + "\n")
    
    print("✅ PROJECT COMPONENTS INSTALLED:\n")
    
    components = [
        ("Configuration", "config.py"),
        ("PyTorch Models", "models/pytorch_models.py"),
        ("TensorFlow Models", "models/tensorflow_models.py"),
        ("Data Preprocessing", "utils/preprocessing.py"),
        ("PyTorch Data Loaders", "utils/torch_data.py"),
        ("Training Utilities", "utils/trainer.py"),
        ("Inference Engine", "utils/inference.py"),
        ("LSTM Training Script", "train_lstm.py"),
        ("CNN Training Script", "train_cnn.py"),
        ("LSTM Inference", "inference_lstm.py"),
        ("CNN Inference", "inference_cnn.py"),
        ("Ensemble Inference", "inference_ensemble.py"),
        ("Interactive Demo", "demo.py"),
        ("Project Documentation", "README.md"),
        ("Quick Start Guide", "QUICKSTART.md"),
    ]
    
    for name, file in components:
        exists = "✓" if Path(file).exists() else "✗"
        print(f"  {exists} {name:.<40} {file}")
    
    print("\n📦 FRAMEWORKS & LIBRARIES:\n")
    print("  ✓ PyTorch - Deep Learning (LSTM, CNN)")
    print("  ✓ TensorFlow/Keras - Alternative implementations")
    print("  ✓ Scikit-Learn - Data preprocessing & utilities")
    print("  ✓ NumPy - Numerical computing")
    print("  ✓ Pandas - Data manipulation")
    print("  ✓ OpenCV - Image processing")
    print("  ✓ Matplotlib - Visualization")
    
    print("\n🏗️ MODELS & ARCHITECTURES:\n")
    print("  ✓ LSTM Predictor (3-layer with Attention)")
    print("  ✓ CNN Chart Classifier (4-block architecture)")
    print("  ✓ CNN Feature Extractor")
    print("  ✓ Hybrid Trading Predictor (Multi-modal)")
    
    print("\n📊 DATA & PREPROCESSING:\n")
    print("  ✓ OHLCV time series processing")
    print("  ✓ Image loading and normalization")
    print("  ✓ MinMax & Standard scaling")
    print("  ✓ Sequence creation for LSTM")
    print("  ✓ Train/Val/Test split")
    print("  ✓ Synthetic data generation")
    
    print("\n🚀 KEY FEATURES:\n")
    print("  ✓ No API keys required")
    print("  ✓ Full source code included")
    print("  ✓ Synthetic data generation")
    print("  ✓ Early stopping and checkpointing")
    print("  ✓ Ensemble learning")
    print("  ✓ Confidence estimation")
    print("  ✓ Batch prediction support")
    print("  ✓ Easy-to-use interfaces")
    
    print("\n" + "="*70 + "\n")


def check_system_info():
    """Display system and library information"""
    print("\n" + "="*70)
    print("SYSTEM & LIBRARY INFORMATION")
    print("="*70 + "\n")
    
    print(f"Python: {__import__('sys').version}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
    except:
        print("TensorFlow: Not installed")
    
    try:
        import sklearn
        print(f"Scikit-Learn: {sklearn.__version__}")
    except:
        print("Scikit-Learn: Not installed")
    
    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n" + "="*70 + "\n")


def clear_checkpoints():
    """Clear saved model checkpoints"""
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print("✓ Checkpoints cleared")
    else:
        print("✓ No checkpoints to clear")


def clear_data():
    """Clear generated data"""
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)
        print("✓ Data cleared")
    else:
        print("✓ No data to clear")


def reset_project():
    """Reset project to clean state"""
    print("\n⚠️  RESETTING PROJECT...\n")
    clear_checkpoints()
    clear_data()
    
    # Recreate directories
    Path("checkpoints").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("data/synthetic_charts").mkdir(exist_ok=True, parents=True)
    
    print("\n✓ Project reset complete\n")


def list_models():
    """List available model files"""
    print("\n" + "="*70)
    print("AVAILABLE MODELS")
    print("="*70 + "\n")
    
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        models = list(checkpoint_dir.glob("*.pt"))
        if models:
            print(f"Found {len(models)} PyTorch models:\n")
            for model in sorted(models):
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"  • {model.name:.<40} ({size_mb:.1f} MB)")
        else:
            print("No models found. Train models with:")
            print("  python train_lstm.py")
            print("  python train_cnn.py")
    else:
        print("Checkpoints directory not found.")
    
    print("\n" + "="*70 + "\n")


def check_data():
    """Check available data"""
    print("\n" + "="*70)
    print("DATA INVENTORY")
    print("="*70 + "\n")
    
    data_dir = Path("data")
    if data_dir.exists():
        charts_dir = data_dir / "synthetic_charts"
        if charts_dir.exists():
            images = list(charts_dir.glob("*.png"))
            print(f"Synthetic chart images: {len(images)}")
        else:
            print("No synthetic charts directory")
    else:
        print("No data directory found")
    
    print("\n" + "="*70 + "\n")


def get_quick_commands():
    """Print quick command reference"""
    print("\n" + "="*70)
    print("QUICK COMMAND REFERENCE")
    print("="*70 + "\n")
    
    commands = {
        "Setup & Info": [
            ("python utils_helper.py --summary", "Project overview"),
            ("python utils_helper.py --system", "System information"),
            ("python utils_helper.py --list-models", "List saved models"),
            ("python utils_helper.py --check-data", "Check available data"),
        ],
        "Training": [
            ("python train_lstm.py", "Train LSTM model"),
            ("python train_cnn.py", "Train CNN model"),
        ],
        "Inference": [
            ("python inference_lstm.py", "LSTM predictions"),
            ("python inference_cnn.py <img>", "CNN predictions"),
            ("python inference_ensemble.py", "Ensemble predictions"),
        ],
        "Interactive": [
            ("python demo.py", "Interactive demo"),
        ],
        "Maintenance": [
            ("python utils_helper.py --reset", "Reset project"),
            ("python utils_helper.py --clear-data", "Clear data"),
        ]
    }
    
    for category, cmds in commands.items():
        print(f"\n{category}:")
        print("-" * 70)
        for cmd, desc in cmds:
            print(f"  {cmd:.<45} {desc}")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main utility function"""
    import sys
    
    if len(sys.argv) < 2:
        get_quick_commands()
        return
    
    command = sys.argv[1].lower()
    
    if command == "--summary":
        print_project_summary()
    elif command == "--system":
        check_system_info()
    elif command == "--list-models":
        list_models()
    elif command == "--check-data":
        check_data()
    elif command == "--reset":
        confirm = input("Are you sure? This will delete all checkpoints and data. (yes/no): ")
        if confirm.lower() == "yes":
            reset_project()
    elif command == "--clear-data":
        clear_data()
    elif command == "--clear-checkpoints":
        clear_checkpoints()
    elif command in ["--help", "-h"]:
        get_quick_commands()
    else:
        print(f"Unknown command: {command}")
        get_quick_commands()


if __name__ == "__main__":
    main()
