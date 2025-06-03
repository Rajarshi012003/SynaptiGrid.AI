"""
Test script for the RL environment module.
This script tests the functionality of the HEMSEnvironment class.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_processor import HEMSDataProcessor
from rl_environment import HEMSEnvironment, OptimizationLayer, visualize_episode

def test_rl_environment():
    """Test the RL environment functionality"""
    print("Testing HEMSEnvironment...")
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data_path = "hems_data_final.csv"
    processor = HEMSDataProcessor(data_path)
    processor.load_data()
    processor.preprocess_data()
    train_data, test_data = processor.split_data(test_size=0.2)
    
    # Create environment
    print("\n2. Creating environment...")
    env = HEMSEnvironment(test_data, episode_length=24)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    print("\n3. Testing environment reset...")
    obs, _ = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Test optimization layer
    print("\n4. Testing optimization layer...")
    opt_layer = OptimizationLayer()
    
    # Create a test action
    test_action = np.array([2.0, 3.0, 2.0])  # HVAC, batt_charge, batt_discharge
    print(f"Original action: {test_action}")
    
    # Apply constraints
    constrained_action = opt_layer.enforce_constraints(obs, test_action)
    print(f"Constrained action: {constrained_action}")
    
    # Test step with random actions
    print("\n5. Testing environment step with random actions...")
    done = False
    step_count = 0
    
    # Store data for visualization
    states = []
    actions = []
    rewards = []
    infos = []
    
    while not done:
        # Generate random action
        action = env.action_space.sample()
        
        # Apply constraints
        action = opt_layer.enforce_constraints(obs, action)
        
        # Take step
        next_obs, reward, done, _, info = env.step(action)
        
        # Store data
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        
        # Update observation
        obs = next_obs
        step_count += 1
        
        print(f"Step {step_count}: Reward = {reward:.4f}")
        print(f"  HVAC = {action[0]:.2f}, Batt_charge = {action[1]:.2f}, Batt_discharge = {action[2]:.2f}")
        print(f"  Temperature = {info['temperature']:.2f}, Battery SOC = {info['battery_soc']:.2f}")
        print(f"  Energy cost = {info['energy_cost']:.4f}, Comfort = {info['comfort_utility']:.4f}")
        print()
    
    # Test episode summary
    print("\n6. Testing episode summary...")
    summary, df = env.get_episode_summary()
    print("Episode summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Visualize episode
    print("\n7. Visualizing episode...")
    visualize_episode(env, df)
    
    # Plot reward components
    plt.figure(figsize=(10, 6))
    
    reward_components = ['cost', 'comfort', 'carbon', 'peak']
    reward_data = np.array([[info['reward_components'][comp] for comp in reward_components] for info in infos])
    
    for i, comp in enumerate(reward_components):
        plt.plot(reward_data[:, i], label=comp)
    
    plt.plot(rewards, label='total', linewidth=2, color='black')
    
    plt.title('Reward Components')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('reward_components.png')
    
    # Test simple rule-based policy
    print("\n8. Testing simple rule-based policy...")
    
    def simple_rule_policy(obs):
        """Simple rule-based policy for testing"""
        temp_in, battery_soc, pv_gen, price, hour, carbon = obs
        
        # HVAC control based on temperature
        if temp_in < 20:
            hvac_power = 2.0
        elif temp_in < 22:
            hvac_power = 1.0
        elif temp_in < 24:
            hvac_power = 0.5
        else:
            hvac_power = 0.0
        
        # Battery control based on PV and price
        if pv_gen > 3.0 and battery_soc < 0.8:
            # Charge from excess PV
            batt_charge = 2.0
            batt_discharge = 0.0
        elif price > 0.15 and battery_soc > 0.3:
            # Discharge during high price
            batt_charge = 0.0
            batt_discharge = 2.0
        else:
            # Default behavior
            batt_charge = 0.0
            batt_discharge = 0.0
        
        return np.array([hvac_power, batt_charge, batt_discharge])
    
    # Run episode with rule-based policy
    obs, _ = env.reset(seed=42)
    done = False
    step_count = 0
    total_reward = 0
    
    while not done:
        # Get action from policy
        action = simple_rule_policy(obs)
        
        # Apply constraints
        action = opt_layer.enforce_constraints(obs, action)
        
        # Take step
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        step_count += 1
    
    # Get episode summary
    rule_summary, rule_df = env.get_episode_summary()
    print("Rule-based policy episode summary:")
    for key, value in rule_summary.items():
        print(f"  {key}: {value}")
    
    print(f"Total reward: {total_reward:.4f}")
    
    # Visualize rule-based episode
    visualize_episode(env, rule_df)
    plt.savefig('rule_based_episode.png')
    
    print("\nAll tests completed successfully!")
    return env

if __name__ == "__main__":
    env = test_rl_environment()
    plt.show()  # Show all plots 