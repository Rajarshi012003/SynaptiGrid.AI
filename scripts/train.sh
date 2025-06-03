#!/bin/bash

# Script to train an optimized HEMS model with SNN integration

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="logs/run_${TIMESTAMP}"

# Make sure directories exist
mkdir -p logs models results

# Print header
echo "================================================"
echo "Starting HEMS Optimized Training with SNN"
echo "Run directory: $RUN_DIR"
echo "Timestamp: $TIMESTAMP"
echo "================================================"

# Run the training script with optimized hyperparameters
python train_optimized.py \
  --timesteps 50000 \
  --episode_length 48 \
  --random_weather true \
  --snn_epochs 15 \
  --snn_lr 0.001 \
  --learning_rate 0.0005 \
  --buffer_size 100000 \
  --batch_size 256 \
  --learning_starts 10000 \
  --gradient_steps 1000 \
  --train_freq 1000 \
  --run_dir "$RUN_DIR"

# Print completion message
echo "================================================"
echo "Training complete!"
echo "Model saved to: $RUN_DIR/td3_hems_best.zip"
echo "SNN model saved to: $RUN_DIR/snn_model.pth"
echo "Run the dashboard to use the new model"
echo "================================================" 