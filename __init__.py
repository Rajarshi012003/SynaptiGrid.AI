"""
Home Energy Management System (HEMS)

A sophisticated energy management system using Spiking Neural Networks and Reinforcement Learning
"""

__version__ = '1.0.0'

# Import core components for easier access
from .rl_environment import HEMSEnvironment, OptimizationLayer
from .rl_agent import train_rl_agent, evaluate_rl_agent, create_real_time_controller
from .snn_model import SNN_Model, train_snn_model, test_snn_model
from .data_processor import HEMSDataProcessor

# Define what gets imported with "from hems import *"
__all__ = [
    'HEMSEnvironment',
    'OptimizationLayer',
    'train_rl_agent',
    'evaluate_rl_agent',
    'create_real_time_controller',
    'SNN_Model',
    'train_snn_model',
    'test_snn_model',
    'HEMSDataProcessor',
] 