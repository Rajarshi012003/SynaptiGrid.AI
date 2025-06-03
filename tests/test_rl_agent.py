"""
Test for RL agent
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  #  non-interactive backend
import matplotlib.pyplot as plt
from data_processor import HEMSDataProcessor
from rl_environment import HEMSEnvironment, OptimizationLayer
from rl_agent import train_rl_agent, evaluate_rl_agent, compare_with_baseline, create_real_time_controller

def test_rl_agent():
    """Test the RL agent functionality with a small number of iterations"""
    print("Testing RL agent...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    os.makedirs("test_logs", exist_ok=True)
    
    print("\n1. Loading and preprocessing data...")
    data_path = "hems_data_final.csv"
    processor = HEMSDataProcessor(data_path)
    processor.load_data()
    processor.preprocess_data()
    train_data, test_data = processor.split_data(test_size=0.2)
    
    print("\n2. Creating RL environments...")
    train_env = HEMSEnvironment(train_data, episode_length=24)
    eval_env = HEMSEnvironment(test_data, episode_length=24)
    
    print("\n3. Training the agent for a small number of timesteps...")
    agent = train_rl_agent(
        train_env, 
        eval_env, 
        total_timesteps=500,  # Small number for testing
        log_dir="test_logs/",
        seed=42
    )
    
    print("\n4. Evaluating the agent...")
    results = evaluate_rl_agent(
        agent, 
        eval_env, 
        n_episodes=1,  # Just one episode for testing
        deterministic=True
    )
    
    print("\n5. Comparing with baseline...")
    comparison = compare_with_baseline(
        results, 
        test_data, 
        log_dir="test_logs/"
    )
    
    print("\n6. Testing real-time controller...")
    controller = create_real_time_controller(agent)
    
    sample_observations = [
        np.array([21.0, 0.5, 3.0, 0.12, 12.0, 200.0]),  # Noon, moderate price, good PV
        np.array([18.0, 0.8, 0.0, 0.25, 20.0, 350.0]),  # Evening, high price, no PV
        np.array([22.0, 0.3, 0.5, 0.08, 6.0, 150.0])    # Morning, low price, low PV
    ]
    
    print("\nTesting real-time controller with sample observations:")
    for i, obs in enumerate(sample_observations):
        action = controller(obs)
        print(f"Sample {i+1}:")
        print(f"  Observation: Temp={obs[0]:.1f}, Battery={obs[1]:.1f}, PV={obs[2]:.1f}, Price=${obs[3]:.2f}, Hour={obs[4]:.0f}, Carbon={obs[5]:.0f}")
        print(f"  Action: HVAC={action[0]:.2f}, Batt_charge={action[1]:.2f}, Batt_discharge={action[2]:.2f}")
    
    print("\n7. Testing model saving and loading...")
    model_path = "test_logs/test_model.zip"
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    from stable_baselines3 import TD3
    loaded_agent = TD3.load(model_path)
    print("Model loaded successfully")
    
    print("\n8. Testing loaded model...")
    for i, obs in enumerate(sample_observations):
        original_action, _ = agent.predict(obs, deterministic=True)
        loaded_action, _ = loaded_agent.predict(obs, deterministic=True)
        
        print(f"Sample {i+1}:")
        print(f"  Original action: {original_action}")
        print(f"  Loaded action: {loaded_action}")
        print(f"  Actions match: {np.allclose(original_action, loaded_action)}")
    
    print("\nAll tests completed successfully!")
    return agent, results, comparison

if __name__ == "__main__":
    agent, results, comparison = test_rl_agent()
    
    if False:
        metrics = ['Energy Cost', 'Comfort', 'Carbon Emissions', 'Peak Demand']
        rl_values = [results['avg_cost'], results['avg_comfort'], 
                    results['avg_emissions'], results['avg_peak']]
        baseline_values = [comparison['baseline']['avg_cost'], comparison['baseline']['avg_comfort'], 
                          comparison['baseline']['avg_emissions'], comparison['baseline']['avg_peak']]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, rl_values, width, label='RL Agent')
        plt.bar(x + width/2, baseline_values, width, label='Baseline')
        
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('RL Agent vs. Baseline Comparison (Test Run)')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('test_comparison.png')
        plt.show() 
