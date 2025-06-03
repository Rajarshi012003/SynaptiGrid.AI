#!/bin/bash

# Dashboard run!!!

echo "================================================"
echo "Starting HEMS Dashboard with Combined RL+SNN Model"
echo "Model: logs/run_20250531_135517/td3_hems_best.zip"
echo "SNN Model: logs/run_20250531_135517/snn_model.pth"
echo "================================================"

# Export environment variables to avoid Qt errors
export QT_QPA_PLATFORM=offscreen

streamlit run hems_dashboard.py 
