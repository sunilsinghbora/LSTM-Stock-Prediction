#!/usr/bin/env python3
"""
Model Evaluation Module for Transformer-based TXN Prediction

This module evaluates the trained transformer model using various metrics
including RMSE, MAE, and directional accuracy.

Usage:
    python src/evaluate.py --checkpoint checkpoints/best.pt
    python src/evaluate.py --config config.yaml --data-dir data
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import yaml
from typing import Dict, Tuple, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# Import the model class from train.py
from train import TransformerPredictor, load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load test data."""
    logger.info(f"Loading test data from: {data_dir}")
    
    X_test = torch.FloatTensor(np.load(os.path.join(data_dir, 'X_test.npy')))
    y_test = torch.FloatTensor(np.load(os.path.join(data_dir, 'y_test.npy')))
    
    logger.info(f"Test data loaded - X: {X_test.shape}, y: {y_test.shape}")
    
    return X_test, y_test


def load_model_checkpoint(checkpoint_path: str, input_dim: int, 
                         config: Dict, seq_len: int) -> TransformerPredictor:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    # Initialize model architecture
    model = TransformerPredictor(
        input_dim=input_dim,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        max_seq_len=seq_len
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded from epoch {checkpoint['epoch']} "
               f"(val_loss: {checkpoint['val_loss']:.6f})")
    
    return model


def get_predictions(model: TransformerPredictor, X_test: torch.Tensor,
                   device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Generate predictions on test set."""
    logger.info("Generating predictions...")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_x = X_test[i:i+batch_size].to(device)
            batch_pred = model(batch_x).squeeze().cpu().numpy()
            
            # Handle single prediction case
            if batch_pred.ndim == 0:
                batch_pred = np.array([batch_pred])
                
            predictions.extend(batch_pred)
    
    predictions = np.array(predictions)
    logger.info(f"Generated {len(predictions)} predictions")
    
    return predictions


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    logger.info("Calculating evaluation metrics...")
    
    metrics = {}
    
    # Regression metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    # Handle zero values by adding small epsilon
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    metrics['mape'] = mape
    
    # Directional accuracy (% of correct sign predictions)
    y_true_sign = np.sign(y_true)
    y_pred_sign = np.sign(y_pred)
    directional_accuracy = np.mean(y_true_sign == y_pred_sign) * 100
    metrics['directional_accuracy'] = directional_accuracy
    
    # Additional statistics
    metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
    metrics['mean_true'] = np.mean(y_true)
    metrics['std_true'] = np.std(y_true)
    metrics['mean_pred'] = np.mean(y_pred)
    metrics['std_pred'] = np.std(y_pred)
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print evaluation metrics in a formatted way."""
    logger.info("=== Model Evaluation Results ===")
    logger.info(f"MSE:                    {metrics['mse']:.8f}")
    logger.info(f"RMSE:                   {metrics['rmse']:.8f}")
    logger.info(f"MAE:                    {metrics['mae']:.8f}")
    logger.info(f"MAPE:                   {metrics['mape']:.2f}%")
    logger.info(f"Directional Accuracy:   {metrics['directional_accuracy']:.2f}%")
    logger.info(f"Correlation:            {metrics['correlation']:.4f}")
    logger.info("=== Data Statistics ===")
    logger.info(f"True mean:              {metrics['mean_true']:.8f}")
    logger.info(f"True std:               {metrics['std_true']:.8f}")
    logger.info(f"Pred mean:              {metrics['mean_pred']:.8f}")
    logger.info(f"Pred std:               {metrics['std_pred']:.8f}")


def create_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray,
                          output_dir: str = 'results') -> None:
    """Create evaluation plots."""
    logger.info("Creating evaluation plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TXN Return Prediction - Model Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted scatter plot
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2, alpha=0.8, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual TXN Returns')
    axes[0, 0].set_ylabel('Predicted TXN Returns')
    axes[0, 0].set_title('Actual vs Predicted Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series plot (last 200 points for clarity)
    n_points = min(200, len(y_true))
    indices = np.arange(-n_points, 0)
    axes[0, 1].plot(indices, y_true[-n_points:], label='Actual', linewidth=1.5)
    axes[0, 1].plot(indices, y_pred[-n_points:], label='Predicted', linewidth=1.5, alpha=0.8)
    axes[0, 1].set_xlabel('Days from End')
    axes[0, 1].set_ylabel('TXN Returns')
    axes[0, 1].set_title(f'Time Series Comparison (Last {n_points} Days)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals plot
    residuals = y_true - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Predicted TXN Returns')
    axes[1, 0].set_ylabel('Residuals (Actual - Predicted)')
    axes[1, 0].set_title('Residuals Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Distribution comparison
    axes[1, 1].hist(y_true, bins=50, alpha=0.7, label='Actual', density=True)
    axes[1, 1].hist(y_pred, bins=50, alpha=0.7, label='Predicted', density=True)
    axes[1, 1].set_xlabel('TXN Returns')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'evaluation_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Evaluation plots saved to: {plot_path}")
    plt.close()
    
    # Create additional directional accuracy plot
    create_directional_accuracy_plot(y_true, y_pred, output_dir)


def create_directional_accuracy_plot(y_true: np.ndarray, y_pred: np.ndarray,
                                   output_dir: str) -> None:
    """Create directional accuracy visualization."""
    
    # Create directional accuracy by return magnitude
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    correct_direction = (true_sign == pred_sign)
    
    # Bin by return magnitude
    abs_returns = np.abs(y_true)
    bins = np.percentile(abs_returns, [0, 25, 50, 75, 100])
    bin_labels = ['Q1 (Smallest)', 'Q2', 'Q3', 'Q4 (Largest)']
    
    bin_accuracies = []
    for i in range(len(bins)-1):
        mask = (abs_returns >= bins[i]) & (abs_returns < bins[i+1])
        if i == len(bins)-2:  # Last bin includes the max value
            mask = (abs_returns >= bins[i]) & (abs_returns <= bins[i+1])
        
        if np.sum(mask) > 0:
            accuracy = np.mean(correct_direction[mask]) * 100
            bin_accuracies.append(accuracy)
        else:
            bin_accuracies.append(0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_labels, bin_accuracies, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random Guess (50%)')
    plt.ylabel('Directional Accuracy (%)')
    plt.xlabel('Return Magnitude Quartiles')
    plt.title('Directional Accuracy by Return Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, bin_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    direction_plot_path = os.path.join(output_dir, 'directional_accuracy.png')
    plt.savefig(direction_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Directional accuracy plot saved to: {direction_plot_path}")
    plt.close()


def save_results(metrics: Dict[str, float], y_true: np.ndarray, y_pred: np.ndarray,
                output_dir: str) -> None:
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to JSON-like format
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("TXN Return Prediction - Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Regression Metrics:\n")
        f.write(f"  MSE:                    {metrics['mse']:.8f}\n")
        f.write(f"  RMSE:                   {metrics['rmse']:.8f}\n")
        f.write(f"  MAE:                    {metrics['mae']:.8f}\n")
        f.write(f"  MAPE:                   {metrics['mape']:.2f}%\n")
        f.write(f"  Correlation:            {metrics['correlation']:.4f}\n\n")
        
        f.write("Classification Metrics:\n")
        f.write(f"  Directional Accuracy:   {metrics['directional_accuracy']:.2f}%\n\n")
        
        f.write("Data Statistics:\n")
        f.write(f"  True Returns - Mean:    {metrics['mean_true']:.8f}\n")
        f.write(f"  True Returns - Std:     {metrics['std_true']:.8f}\n")
        f.write(f"  Pred Returns - Mean:    {metrics['mean_pred']:.8f}\n")
        f.write(f"  Pred Returns - Std:     {metrics['std_pred']:.8f}\n")
    
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'actual_returns': y_true,
        'predicted_returns': y_pred,
        'residuals': y_true - y_pred,
        'abs_error': np.abs(y_true - y_pred),
        'correct_direction': np.sign(y_true) == np.sign(y_pred)
    })
    
    results_path = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Detailed predictions saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained transformer model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing test data')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load test data
        X_test, y_test = load_test_data(args.data_dir)
        
        # Get input dimensions
        input_dim = X_test.shape[-1]
        seq_len = X_test.shape[1]
        
        # Load model
        model = load_model_checkpoint(args.checkpoint, input_dim, config, seq_len)
        model.to(device)
        
        # Generate predictions
        y_pred = get_predictions(model, X_test, device, 
                               batch_size=config['training']['batch_size'])
        y_true = y_test.numpy()
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Print results
        print_metrics(metrics)
        
        # Create plots
        create_evaluation_plots(y_true, y_pred, args.output_dir)
        
        # Save results
        save_results(metrics, y_true, y_pred, args.output_dir)
        
        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Summary statistics
        logger.info(f"\n=== SUMMARY ===")
        logger.info(f"RMSE: {metrics['rmse']:.6f}")
        logger.info(f"MAE: {metrics['mae']:.6f}")
        logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()