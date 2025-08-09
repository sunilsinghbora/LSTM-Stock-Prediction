#!/usr/bin/env python3
"""
Transformer Model Training Module for Cross-Asset TXN Prediction

This module implements and trains a PyTorch Transformer encoder model
to predict next-day TXN log returns based on multivariate time series
features from multiple large-cap stocks.

Usage:
    python src/train.py --config config.yaml
    python src/train.py --data-dir data --checkpoint-dir checkpoints
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
from typing import Dict, Tuple
import logging
from tqdm import tqdm
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer model for time series prediction.
    
    Architecture:
    1. Linear projection of features -> d_model
    2. Positional encoding  
    3. Stacked TransformerEncoder layers
    4. Mean pooling over sequence
    5. MLP head to predict next-day TXN log return
    """
    
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 1024,
                 dropout: float = 0.1, max_seq_len: int = 100):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # Expected shape: [seq_len, batch, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head (MLP)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single output for TXN return prediction
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tensor of shape [batch_size, 1] with predicted returns
        """
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        
        # Project to d_model
        # x shape: [batch_size, seq_len, d_model]
        x = self.input_projection(x)
        
        # Transpose for transformer: [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Mean pooling over sequence dimension
        # transformer_out shape: [seq_len, batch_size, d_model]
        pooled = torch.mean(transformer_out, dim=0)  # [batch_size, d_model]
        
        # Pass through output head
        output = self.output_head(pooled)  # [batch_size, 1]
        
        return output


def load_processed_data(data_dir: str) -> Tuple[torch.Tensor, ...]:
    """Load processed data arrays."""
    logger.info(f"Loading processed data from: {data_dir}")
    
    X_train = torch.FloatTensor(np.load(os.path.join(data_dir, 'X_train.npy')))
    X_val = torch.FloatTensor(np.load(os.path.join(data_dir, 'X_val.npy')))
    X_test = torch.FloatTensor(np.load(os.path.join(data_dir, 'X_test.npy')))
    
    y_train = torch.FloatTensor(np.load(os.path.join(data_dir, 'y_train.npy')))
    y_val = torch.FloatTensor(np.load(os.path.join(data_dir, 'y_val.npy')))
    y_test = torch.FloatTensor(np.load(os.path.join(data_dir, 'y_test.npy')))
    
    logger.info(f"Data loaded - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_loaders(X_train: torch.Tensor, X_val: torch.Tensor,
                       y_train: torch.Tensor, y_val: torch.Tensor,
                       batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders."""
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


def train_epoch(model: nn.Module, train_loader: DataLoader,
               criterion: nn.Module, optimizer: optim.Optimizer,
               device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in tqdm(train_loader, desc="Training"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x).squeeze()
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(model: nn.Module, val_loader: DataLoader,
                  criterion: nn.Module, device: torch.device) -> float:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                   epoch: int, train_loss: float, val_loss: float,
                   checkpoint_dir: str, is_best: bool = False) -> str:
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    if is_best:
        filename = 'best.pt'
    else:
        filename = f'checkpoint_epoch_{epoch}.pt'
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    
    return filepath


def train_model(config: Dict, data_dir: str, checkpoint_dir: str) -> str:
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(data_dir)
    
    # Get input dimensions
    input_dim = X_train.shape[-1]  # Number of features
    seq_len = X_train.shape[1]     # Sequence length
    
    logger.info(f"Input dimensions: {input_dim} features, {seq_len} sequence length")
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader, val_loader = create_data_loaders(
        X_train, X_val, y_train, y_val, batch_size)
    
    # Initialize model
    model = TransformerPredictor(
        input_dim=input_dim,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        max_seq_len=seq_len
    ).to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'])
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    best_checkpoint_path = None
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_checkpoint_path = save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                checkpoint_dir, is_best=True)
            logger.info(f"New best model saved: {best_checkpoint_path}")
        
        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                           checkpoint_dir, is_best=False)
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    return best_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Train transformer model for TXN prediction')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing processed data')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        # Train model
        best_model_path = train_model(config, args.data_dir, args.checkpoint_dir)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Best model saved at: {best_model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()