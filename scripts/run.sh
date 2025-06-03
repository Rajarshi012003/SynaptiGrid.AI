#!/bin/bash

# Run the HEMS dashboard with the best model

echo "================================================"
echo "Starting HEMS Dashboard with Combined RL+SNN Model"
echo "Model: logs/run_20250531_135517/td3_hems_best.zip"
echo "SNN Model: logs/run_20250531_135517/snn_model.pth"
echo "================================================"

# Export environment variables to avoid Qt errors
export QT_QPA_PLATFORM=offscreen

# Run the dashboard
streamlit run hems_dashboard.py 