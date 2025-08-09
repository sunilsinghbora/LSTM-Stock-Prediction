#!/usr/bin/env python3
"""
Demo script to run the complete Transformer-based TXN prediction pipeline.

This script demonstrates the full workflow from data generation to model evaluation.
For production use, replace the synthetic data generation with real Yahoo Finance data.

Usage:
    python demo.py
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ ERROR")
        print(result.stderr)
        return False
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 Transformer-based Cross-Asset TXN Prediction                 ║
║                              Demo Pipeline                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

This demo will run the complete pipeline:
1. Generate synthetic stock data (for demo purposes)
2. Engineer features from multiple assets
3. Train a Transformer model
4. Evaluate the model performance
5. Generate visualizations

In production, replace step 1 with: python src/download_data.py
""")

    # Step 1: Generate sample data for demo
    if not run_command(
        "python /tmp/create_sample_data.py", 
        "Step 1: Generate synthetic stock data"
    ):
        return
    
    # Step 2: Feature engineering
    if not run_command(
        "python src/features.py --config config.yaml",
        "Step 2: Feature engineering and preprocessing"
    ):
        return
        
    # Step 3: Train model
    if not run_command(
        "python src/train.py --config config.yaml",
        "Step 3: Train Transformer model"
    ):
        return
        
    # Step 4: Evaluate model
    if not run_command(
        "python src/evaluate.py --checkpoint checkpoints/best.pt",
        "Step 4: Evaluate model performance"
    ):
        return
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            🎉 DEMO COMPLETED! 🎉                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Results have been generated in the following locations:

📊 Model Performance:
   - results/evaluation_metrics.txt    # Detailed metrics
   - results/predictions.csv           # Prediction data
   
📈 Visualizations:
   - results/evaluation_results.png    # Comprehensive evaluation plots
   - results/directional_accuracy.png  # Direction prediction analysis
   
💾 Model Artifacts:
   - checkpoints/best.pt              # Trained model
   - data/feature_scaler.pkl          # Feature scaler
   
🛠️ Configuration:
   - config.yaml                      # Model and training settings

To use with real data:
1. Run: python src/download_data.py --start 2015-01-01 --end 2025-01-01
2. Then follow steps 2-4 above

For more details, see README.md
""")

if __name__ == "__main__":
    main()