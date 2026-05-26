"""
Training script for CNN image classification model
Usage: python train_cnn.py
"""

import torch
import numpy as np
from pathlib import Path

from config import ModelConfig
from models.pytorch_models import CNNChartClassifier
from utils.torch_data import ChartImageDataset, DataLoader as TorchDataLoader
from utils.trainer import CNNTrainer


def generate_synthetic_chart_dataset(num_samples: int = 200):
    """Generate synthetic chart images for testing"""
    import cv2
    from utils.preprocessing import ImagePreprocessor
    
    data_dir = Path("data/synthetic_charts")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic chart images...")
    
    image_paths = []
    labels = []
    
    for i in range(num_samples):
        # Generate random chart pattern (simulated)
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        # Create simple patterns for each class
        class_label = i % 3  # 0: down, 1: neutral, 2: up
        
        if class_label == 0:  # Downtrend
            for j in range(0, 224, 10):
                cv2.line(img, (j, 50), (j+5, 100), (255, 0, 0), 2)
        elif class_label == 1:  # Neutral
            for j in range(0, 224, 10):
                cv2.line(img, (j, 112), (j+10, 112), (0, 255, 0), 2)
        else:  # Uptrend
            for j in range(0, 224, 10):
                cv2.line(img, (j, 200), (j+5, 100), (0, 0, 255), 2)
        
        # Save image
        img_path = data_dir / f"chart_{i:04d}.png"
        cv2.imwrite(str(img_path), img)
        image_paths.append(str(img_path))
        labels.append(class_label)
    
    return image_paths, np.array(labels)


def train_cnn_model():
    """Train CNN model on synthetic chart images"""
    
    print("="*60)
    print("CNN CHART IMAGE CLASSIFIER - TRAINING")
    print("="*60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = ModelConfig.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")
    
    # Generate synthetic data
    print("Generating synthetic chart images...")
    image_paths, labels = generate_synthetic_chart_dataset(num_samples=200)
    print(f"Generated {len(image_paths)} images")
    
    # Train/Val/Test split
    print("\nSplitting data...")
    n = len(image_paths)
    train_end = int(n * ModelConfig.TRAIN_SPLIT)
    val_end = train_end + int(n * ModelConfig.VAL_SPLIT)
    
    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, n)
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_dataset = ChartImageDataset(
        [image_paths[i] for i in train_indices],
        labels[train_indices]
    )
    val_dataset = ChartImageDataset(
        [image_paths[i] for i in val_indices],
        labels[val_indices]
    )
    test_dataset = ChartImageDataset(
        [image_paths[i] for i in test_indices],
        labels[test_indices]
    )
    
    train_loader = TorchDataLoader(train_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=ModelConfig.BATCH_SIZE)
    test_loader = TorchDataLoader(test_dataset, batch_size=ModelConfig.BATCH_SIZE)
    
    # Create model
    print("\nCreating CNN model...")
    model = CNNChartClassifier(
        num_channels=ModelConfig.CNN_NUM_CHANNELS,
        num_classes=ModelConfig.CNN_NUM_CLASSES,
        dropout=ModelConfig.CNN_DROPOUT
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = CNNTrainer(model, device=device, learning_rate=ModelConfig.LEARNING_RATE)
    
    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING...")
    print("="*60 + "\n")
    
    history = trainer.train(
        train_loader, val_loader,
        epochs=ModelConfig.EPOCHS,
        checkpoint_dir="checkpoints"
    )
    
    # Test
    print("\n" + "="*60)
    print("TESTING MODEL...")
    print("="*60 + "\n")
    
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_checkpoint("checkpoints/cnn_final.pt", ModelConfig.EPOCHS, test_loss)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return history, test_loss, test_acc


if __name__ == "__main__":
    history, test_loss, test_acc = train_cnn_model()
    
    print("\nTraining completed successfully!")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print("\nCheckpoints saved in 'checkpoints/' directory")
    print("Use 'inference_cnn.py' to make predictions with the trained model")
