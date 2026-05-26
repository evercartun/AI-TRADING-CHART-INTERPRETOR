"""
Inference script for CNN model - predict trading trends from chart images
Usage: python inference_cnn.py <image_path>
"""

import sys
from pathlib import Path

from utils.preprocessing import ImagePreprocessor
from utils.inference import CNNInferencer, create_prediction_summary


def demo_cnn_prediction():
    """Demonstrate CNN inference"""
    
    print("="*60)
    print("CNN CHART CLASSIFIER - INFERENCE DEMO")
    print("="*60 + "\n")
    
    # Check if model exists
    model_path = Path("checkpoints/cnn_final.pt")
    if not model_path.exists():
        print("ERROR: Model not found at checkpoints/cnn_final.pt")
        print("Please run 'python train_cnn.py' first to train the model.\n")
        return
    
    print("Loading model...\n")
    inferencer = CNNInferencer(str(model_path), device="cpu")
    
    # Look for test images
    test_images_dir = Path("data/synthetic_charts")
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.png"))[:5]
        
        if test_images:
            print(f"Found {len(test_images)} test images\n")
            
            for img_path in test_images:
                print(f"Analyzing: {img_path.name}")
                
                try:
                    class_name, confidence = inferencer.predict(str(img_path))
                    scores = inferencer.predict_with_all_scores(str(img_path))
                    
                    result = {
                        'cnn_prediction': class_name,
                        'cnn_confidence': confidence,
                        'cnn_scores': scores
                    }
                    
                    print(f"  Prediction: {class_name}")
                    print(f"  Confidence: {confidence:.2%}")
                    print(f"  Scores: {scores}\n")
                
                except Exception as e:
                    print(f"  Error: {e}\n")
        else:
            print("No test images found in data/synthetic_charts/")
    else:
        print("No synthetic charts directory found.")
        print("Run 'python train_cnn.py' to generate test images.\n")


def predict_single_image(image_path: str):
    """Predict on a single image"""
    
    print("="*60)
    print("CNN CHART CLASSIFIER - SINGLE IMAGE PREDICTION")
    print("="*60 + "\n")
    
    model_path = Path("checkpoints/cnn_final.pt")
    if not model_path.exists():
        print("ERROR: Model not found at checkpoints/cnn_final.pt")
        print("Train the model first with 'python train_cnn.py'\n")
        return
    
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"ERROR: Image not found at {image_path}\n")
        return
    
    print(f"Loading model...\n")
    inferencer = CNNInferencer(str(model_path), device="cpu")
    
    print(f"Analyzing image: {image_path}\n")
    
    try:
        class_name, confidence = inferencer.predict(image_path)
        scores = inferencer.predict_with_all_scores(image_path)
        
        result = {
            'cnn_prediction': class_name,
            'cnn_confidence': confidence,
            'cnn_scores': scores
        }
        
        summary = create_prediction_summary(result)
        print(summary)
        
    except Exception as e:
        print(f"Error during prediction: {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Predict on specific image
        image_path = sys.argv[1]
        predict_single_image(image_path)
    else:
        # Run demo
        demo_cnn_prediction()
        
        print("\n" + "="*60)
        print("To predict on a specific chart image:")
        print("  python inference_cnn.py <path_to_image>")
        print("="*60 + "\n")
