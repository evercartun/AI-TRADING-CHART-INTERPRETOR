"""
Inference and prediction utilities for trading models
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List
import cv2

from utils.preprocessing import DataPreprocessor, ImagePreprocessor
from models.pytorch_models import LSTMPredictor, CNNChartClassifier
from config import ModelConfig, InferenceConfig


class LSTMInferencer:
    """Inference engine for LSTM time series models"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize inferencer
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.model = LSTMPredictor().to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        self.preprocessor = DataPreprocessor()
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make prediction on time series data
        
        Args:
            data: Time series array of shape (seq_length, features)
                 Expected features: [open, high, low, close, volume]
            
        Returns:
            Predicted price change
        """
        # Normalize
        normalized, _ = self.preprocessor.normalize_data(data)
        
        # Convert to tensor
        X = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(X).cpu().numpy()
        
        # Denormalize
        pred_denorm = self.preprocessor.denormalize_data(prediction)
        
        return pred_denorm.squeeze()
    
    def predict_batch(self, data_list: List[np.ndarray]) -> np.ndarray:
        """
        Make predictions on batch of sequences
        
        Args:
            data_list: List of time series arrays
            
        Returns:
            Batch of predictions
        """
        predictions = []
        for data in data_list:
            pred = self.predict(data)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_with_confidence(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Make prediction with confidence estimate
        
        Args:
            data: Time series data
            
        Returns:
            (prediction, confidence_score)
        """
        # Make multiple predictions with slight perturbations for uncertainty estimation
        predictions = []
        num_samples = 5
        
        for _ in range(num_samples):
            # Add small noise
            noisy_data = data + np.random.normal(0, 0.01, data.shape)
            pred = self.predict(noisy_data)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Confidence is inverse of standard deviation
        confidence = 1.0 / (1.0 + std_pred)
        
        return mean_pred, confidence


class CNNInferencer:
    """Inference engine for CNN image classification models"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize inferencer
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.model = CNNChartClassifier().to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        self.class_names = {0: "Downtrend", 1: "Neutral", 2: "Uptrend"}
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Classify chart image
        
        Args:
            image_path: Path to chart image
            
        Returns:
            (class_name, confidence_score)
        """
        # Load and preprocess image
        image = ImagePreprocessor.load_and_preprocess_image(
            image_path,
            target_size=(224, 224)
        )
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(np.transpose(image, (2, 0, 1))).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        class_name = self.class_names[pred_class]
        
        return class_name, confidence
    
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[str, float]]:
        """
        Classify batch of chart images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of (class_name, confidence) tuples
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)
        
        return results
    
    def predict_with_all_scores(self, image_path: str) -> dict:
        """
        Get prediction scores for all classes
        
        Args:
            image_path: Path to chart image
            
        Returns:
            Dictionary with class scores
        """
        image = ImagePreprocessor.load_and_preprocess_image(
            image_path,
            target_size=(224, 224)
        )
        
        image_tensor = torch.FloatTensor(np.transpose(image, (2, 0, 1))).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
        
        scores = {
            "downtrend": float(probs[0, 0].item()),
            "neutral": float(probs[0, 1].item()),
            "uptrend": float(probs[0, 2].item()),
        }
        
        return scores


class EnsemblePredictor:
    """Ensemble predictor combining multiple models"""
    
    def __init__(self, lstm_model_path: Optional[str] = None,
                 cnn_model_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize ensemble
        
        Args:
            lstm_model_path: Path to LSTM model (optional)
            cnn_model_path: Path to CNN model (optional)
            device: Device to run on
        """
        self.device = device
        
        self.lstm_inferencer = None
        if lstm_model_path:
            self.lstm_inferencer = LSTMInferencer(lstm_model_path, device)
        
        self.cnn_inferencer = None
        if cnn_model_path:
            self.cnn_inferencer = CNNInferencer(cnn_model_path, device)
    
    def predict_combined(self, ohlcv_data: np.ndarray,
                        image_path: Optional[str] = None) -> dict:
        """
        Make prediction using both models (if available)
        
        Args:
            ohlcv_data: Time series data for LSTM
            image_path: Path to chart image for CNN (optional)
            
        Returns:
            Dictionary with combined predictions
        """
        results = {}
        
        # LSTM prediction
        if self.lstm_inferencer:
            lstm_pred, lstm_conf = self.lstm_inferencer.predict_with_confidence(ohlcv_data)
            results['lstm_prediction'] = float(lstm_pred)
            results['lstm_confidence'] = float(lstm_conf)
        
        # CNN prediction
        if self.cnn_inferencer and image_path:
            cnn_class, cnn_conf = self.cnn_inferencer.predict(image_path)
            results['cnn_prediction'] = cnn_class
            results['cnn_confidence'] = float(cnn_conf)
            results['cnn_scores'] = self.cnn_inferencer.predict_with_all_scores(image_path)
        
        # Combined decision
        if self.lstm_inferencer and self.cnn_inferencer and image_path:
            lstm_signal = "up" if results['lstm_prediction'] > 0 else "down"
            cnn_signal = "up" if results['cnn_prediction'] == "Uptrend" else \
                        "down" if results['cnn_prediction'] == "Downtrend" else "neutral"
            
            results['ensemble_signal'] = self._combine_signals(lstm_signal, cnn_signal)
            results['combined_confidence'] = (results['lstm_confidence'] + results['cnn_confidence']) / 2
        
        return results
    
    @staticmethod
    def _combine_signals(lstm_signal: str, cnn_signal: str) -> str:
        """Combine signals from multiple models"""
        if lstm_signal == cnn_signal:
            return lstm_signal
        elif lstm_signal == "neutral" or cnn_signal == "neutral":
            return "neutral"
        else:
            return "conflicting"


def create_prediction_summary(prediction_result: dict) -> str:
    """
    Create a human-readable prediction summary
    
    Args:
        prediction_result: Dictionary from ensemble predictor
        
    Returns:
        Formatted summary string
    """
    summary = "="*50 + "\n"
    summary += "TRADING PREDICTION SUMMARY\n"
    summary += "="*50 + "\n\n"
    
    if 'lstm_prediction' in prediction_result:
        summary += f"LSTM Time Series Model:\n"
        summary += f"  Prediction: {prediction_result['lstm_prediction']:.4f}\n"
        summary += f"  Confidence: {prediction_result['lstm_confidence']:.2%}\n\n"
    
    if 'cnn_prediction' in prediction_result:
        summary += f"CNN Chart Analysis Model:\n"
        summary += f"  Prediction: {prediction_result['cnn_prediction']}\n"
        summary += f"  Confidence: {prediction_result['cnn_confidence']:.2%}\n"
        summary += f"  Scores: {prediction_result['cnn_scores']}\n\n"
    
    if 'ensemble_signal' in prediction_result:
        summary += f"ENSEMBLE DECISION:\n"
        summary += f"  Signal: {prediction_result['ensemble_signal'].upper()}\n"
        summary += f"  Combined Confidence: {prediction_result['combined_confidence']:.2%}\n"
    
    summary += "="*50 + "\n"
    
    return summary
