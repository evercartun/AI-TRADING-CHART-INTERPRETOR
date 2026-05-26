"""
Setup verification and initial configuration script
Run this to verify your installation is complete
"""

import sys
from pathlib import Path


def check_files():
    """Check if all required files exist"""
    print("\n" + "="*70)
    print("CHECKING PROJECT FILES")
    print("="*70 + "\n")
    
    required_files = [
        ("config.py", "Configuration"),
        ("requirements.txt", "Dependencies"),
        ("demo.py", "Demo script"),
        ("README.md", "Documentation"),
        ("models/pytorch_models.py", "PyTorch models"),
        ("models/tensorflow_models.py", "TensorFlow models"),
        ("utils/preprocessing.py", "Data preprocessing"),
        ("utils/trainer.py", "Training utilities"),
        ("utils/inference.py", "Inference engines"),
        ("train_lstm.py", "LSTM training"),
        ("train_cnn.py", "CNN training"),
        ("inference_lstm.py", "LSTM inference"),
        ("inference_cnn.py", "CNN inference"),
        ("inference_ensemble.py", "Ensemble inference"),
    ]
    
    all_exist = True
    for file, description in required_files:
        exists = Path(file).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file:.<50} {description}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_packages():
    """Check if required packages are installed"""
    print("\n" + "="*70)
    print("CHECKING INSTALLED PACKAGES")
    print("="*70 + "\n")
    
    packages = [
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("sklearn", "Scikit-Learn"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib"),
    ]
    
    missing = []
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(name)
    
    return len(missing) == 0, missing


def check_directories():
    """Check if required directories exist"""
    print("\n" + "="*70)
    print("CHECKING DIRECTORIES")
    print("="*70 + "\n")
    
    directories = [
        ("models", "Models directory"),
        ("utils", "Utils directory"),
        ("data", "Data directory"),
        ("checkpoints", "Checkpoints directory"),
        ("notebooks", "Notebooks directory"),
    ]
    
    for dir_path, description in directories:
        path = Path(dir_path)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path:.<40} {description}")


def print_next_steps():
    """Print next steps"""
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70 + "\n")
    
    print("1. Install dependencies (if not done):")
    print("   pip install -r requirements.txt\n")
    
    print("2. View project summary:")
    print("   python BUILD_SUMMARY.py\n")
    
    print("3. Run interactive demo:")
    print("   python demo.py\n")
    
    print("4. Train models:")
    print("   python train_lstm.py")
    print("   python train_cnn.py\n")
    
    print("5. Make predictions:")
    print("   python inference_lstm.py")
    print("   python inference_cnn.py")
    print("   python inference_ensemble.py\n")


def main():
    """Run all checks"""
    print("\n" + "="*70)
    print("AI TRADING CHART PREDICTOR - SETUP VERIFICATION")
    print("="*70)
    
    # Check files
    files_ok = check_files()
    
    # Check directories
    check_directories()
    
    # Check packages
    packages_ok, missing = check_packages()
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70 + "\n")
    
    if files_ok and packages_ok:
        print("✓ All checks passed! Your installation is complete.\n")
        print_next_steps()
        return 0
    else:
        print("⚠️  Some issues detected:\n")
        if not files_ok:
            print("  - Some project files are missing")
        if not packages_ok:
            print(f"  - Missing packages: {', '.join(missing)}")
            print("\n  Install missing packages with:")
            print("  pip install -r requirements.txt\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
