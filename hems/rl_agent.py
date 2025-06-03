"""
Reinforcement Learning Agent for HEMS
"""

import os
import numpy as np
import torch
import torch.nn as nn
import gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
from rl_environment import HEMSEnvironment, OptimizationLayer, visualize_episode, BATTERY_CAPACITY
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for the TD3 policy network"""
    
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        n_input = int(np.prod(observation_space.shape))
        
        self.feature_net = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.LayerNorm(128),  # stability
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        return self.feature_net(observations)


class CurricularTrainingCallback(BaseCallback):
    """Callback for implementing curricular learning during training"""
    
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.episode_count = 0
        self.current_difficulty = 0
        self.max_difficulty = 3
        self.episodes_per_level = 50
    
    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        """Update difficulty level at the end of each rollout"""
        self.episode_count += 1
        
        # Increase difficulty every episodes_per_level episodes
        if self.episode_count % self.episodes_per_level == 0 and self.current_difficulty < self.max_difficulty:
            self.current_difficulty += 1
            
            # Update environment parameters based on difficulty
            if self.current_difficulty == 1:
                # Level 1: Add mild randomization
                self.env.random_weather = True
            elif self.current_difficulty == 2:
                # Level 2: Increase challenge with more price variations(in env)
                pass
            elif self.current_difficulty == 3:
                # Level 3: Full challenge with all variations(in env)
                pass
            
            if self.verbose > 0:
                print(f"Increasing training difficulty to level {self.current_difficulty}")


class EpisodeLogCallback(BaseCallback):
    """Callback for logging episode information during training"""
    
    def __init__(self, eval_env, log_dir='./logs/', verbose=0, eval_freq=10):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.episode_count = 0
        self.eval_freq = eval_freq
        self.best_reward = -float('inf')
        
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_comfort = []
        self.episode_emissions = []
        self.episode_peaks = []
        self.episode_battery_usage = []
        
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        """Called after a rollout is completed"""
        self.episode_count += 1
        
        if self.episode_count % self.eval_freq == 0:  # Evaluate every eval_freq episodes
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.eval_env.step(action)
                episode_reward += reward
            
            summary, df = self.eval_env.get_episode_summary()
            
            reward_components = self.eval_env.history[-1]['reward_components']
            
            self.episode_rewards.append(episode_reward)
            self.episode_costs.append(summary['total_energy_cost'])
            self.episode_comfort.append(summary['average_comfort'])
            self.episode_emissions.append(summary['total_carbon_emissions'])
            self.episode_peaks.append(summary['peak_demand'])
            
            battery_discharge_total = df['batt_discharge'].sum()
            battery_charge_total = df['batt_charge'].sum()
            self.episode_battery_usage.append(battery_discharge_total + battery_charge_total)
            
            log_df = pd.DataFrame({
                'episode': [self.episode_count],
                'reward': [episode_reward],
                'energy_cost': [summary['total_energy_cost']],
                'comfort': [summary['average_comfort']],
                'emissions': [summary['total_carbon_emissions']],
                'peak_demand': [summary['peak_demand']],
                'battery_usage': [battery_discharge_total + battery_charge_total],
                'cost_component': [reward_components['cost']],
                'comfort_component': [reward_components['comfort']],
                'carbon_component': [reward_components['carbon']],
                'peak_component': [reward_components['peak']],
                'battery_component': [reward_components.get('battery', 0)]
            })
            
            log_path = os.path.join(self.log_dir, 'training_log.csv')
            if os.path.exists(log_path):
                log_df.to_csv(log_path, mode='a', header=False, index=False)
            else:
                log_df.to_csv(log_path, mode='w', header=True, index=False)
            
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.model.save(os.path.join(self.log_dir, "td3_hems_best"))
                if self.verbose > 0:
                    print(f"New best model saved with reward: {episode_reward:.2f}")
            
            if self.episode_count % 50 == 0:
                self._plot_training_progress()
                
                self.model.save(os.path.join(self.log_dir, f"td3_hems_{self.episode_count}"))
    
    def _plot_training_progress(self):
        """Plot training progress"""
        figures_dir = os.path.join(self.log_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        
        axs[0, 0].plot(self.episode_rewards)
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Evaluation Episode')
        axs[0, 0].set_ylabel('Average Reward')
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(self.episode_costs)
        axs[0, 1].set_title('Energy Costs')
        axs[0, 1].set_xlabel('Evaluation Episode')
        axs[0, 1].set_ylabel('Total Cost')
        axs[0, 1].grid(True)
        
        axs[1, 0].plot(self.episode_comfort)
        axs[1, 0].set_title('Comfort')
        axs[1, 0].set_xlabel('Evaluation Episode')
        axs[1, 0].set_ylabel('Average Comfort')
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(self.episode_emissions)
        axs[1, 1].set_title('Carbon Emissions')
        axs[1, 1].set_xlabel('Evaluation Episode')
        axs[1, 1].set_ylabel('Total Emissions')
        axs[1, 1].grid(True)
        
        axs[2, 0].plot(self.episode_peaks)
        axs[2, 0].set_title('Peak Demand')
        axs[2, 0].set_xlabel('Evaluation Episode')
        axs[2, 0].set_ylabel('Peak Demand (kW)')
        axs[2, 0].grid(True)
        
        axs[2, 1].plot(self.episode_battery_usage)
        axs[2, 1].set_title('Battery Usage')
        axs[2, 1].set_xlabel('Evaluation Episode')
        axs[2, 1].set_ylabel('Total kWh')
        axs[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'training_progress_{self.episode_count}.png'))
        plt.close()


def train_rl_agent(train_env, eval_env, total_timesteps=200000, log_dir='./logs/', seed=42, callbacks=None,
                learning_rate=1e-4, buffer_size=100000, learning_starts=5000, batch_size=512, 
                train_freq=1, gradient_steps=1):
    """Train a TD3 agent for HEMS control"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_env = Monitor(train_env)
    
    policy_kwargs = {
        "net_arch": {
            "pi": [256, 256],  # Actor network
            "qf": [256, 256]   # Critic network
        },
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128}
    }
    
    # Define action noise for exploration
    n_actions = train_env.action_space.shape[0]
    
    # Use Ornstein-Uhlenbeck noise with reduced parameters for more stable exploration
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.2 * np.ones(n_actions),  
        theta=0.1  
    )
    
    model = TD3(
        "MlpPolicy",
        train_env,
        action_noise=action_noise,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=0.002,  
        gamma=0.99,
        policy_delay=2,
        target_policy_noise=0.1,  
        target_noise_clip=0.2, 
        verbose=1,
        seed=seed,
        device='auto',
        policy_kwargs=policy_kwargs,
        gradient_steps=gradient_steps,
        train_freq=train_freq,
        tensorboard_log=os.path.join(log_dir, "tb_logs")
    )
    
    if callbacks is None:
        eval_callback = EpisodeLogCallback(eval_env, log_dir=log_dir, verbose=1, eval_freq=10)
        curricular_callback = CurricularTrainingCallback(train_env, verbose=1)
        callbacks = CallbackList([eval_callback, curricular_callback])
    
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    
    model.save(os.path.join(log_dir, "td3_hems_final"))
    
    print(f"Training completed. Model saved to {log_dir}/td3_hems_final")
    
    return model


def evaluate_rl_agent(agent, env, n_episodes=5, deterministic=True):
    """Evaluate the performance of a trained agent"""
    # Create optimization layer for constraint enforcement
    opt_layer = OptimizationLayer()
    
    all_rewards = []
    all_summaries = []
    all_dfs = []
    
    for episode in range(n_episodes):
        print(f"Starting evaluation episode {episode+1}/{n_episodes}")
        
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            #  action from model
            action_rl, _ = agent.predict(obs, deterministic=deterministic)
            
            # Apply safety layer to enforce constraints
            action = opt_layer.enforce_constraints(obs, action_rl)
            
            # Take step in environment
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
        
        summary, df = env.get_episode_summary()
        
        try:
            visualize_episode(env, df)
        except:
            print("Warning: Could not visualize episode")
        
        all_rewards.append(episode_reward)
        all_summaries.append(summary)
        all_dfs.append(df)
        
        print(f"Episode {episode+1} completed. Total reward: {episode_reward:.2f}")
        print(f"Total energy cost: {summary['total_energy_cost']:.2f}")
        print(f"Average comfort: {summary['average_comfort']:.2f}")
        print(f"Total carbon emissions: {summary['total_carbon_emissions']:.2f}")
        print(f"Peak demand: {summary['peak_demand']:.2f}")
        print(f"Battery cycles: {(df['batt_charge'].sum() + df['batt_discharge'].sum()) / (2 * BATTERY_CAPACITY):.2f}")
        print("-" * 50)
    
    avg_reward = np.mean(all_rewards)
    avg_cost = np.mean([s['total_energy_cost'] for s in all_summaries])
    avg_comfort = np.mean([s['average_comfort'] for s in all_summaries])
    avg_emissions = np.mean([s['total_carbon_emissions'] for s in all_summaries])
    avg_peak = np.mean([s['peak_demand'] for s in all_summaries])
    
    print("\nEvaluation Summary:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average energy cost: {avg_cost:.2f}")
    print(f"Average comfort: {avg_comfort:.2f}")
    print(f"Average carbon emissions: {avg_emissions:.2f}")
    print(f"Average peak demand: {avg_peak:.2f}")
    
    results = {
        'rewards': all_rewards,
        'summaries': all_summaries,
        'dfs': all_dfs,
        'avg_reward': avg_reward,
        'avg_cost': avg_cost,
        'avg_comfort': avg_comfort,
        'avg_emissions': avg_emissions,
        'avg_peak': avg_peak
    }
    
    return results


def compare_with_baseline(rl_results, test_data, log_dir='./logs/'):
    """Compare RL agent performance with rule-based baseline"""
    
    def rule_based_policy(obs):
        """Simple rule-based policy for comparison"""
        temp_in, battery_soc, pv_gen, price, hour, carbon = obs
        
        hvac_power = 0.0
        batt_charge = 0.0
        batt_discharge = 0.0
        
        if temp_in < 18.0:
            hvac_power = 2.0
        elif temp_in > 25.0:
            hvac_power = 1.5
        
        if price < 0.1:  
           
            if battery_soc < 0.9:
                batt_charge = min(5.0, pv_gen + 1.0)  
                batt_discharge = 0.0
        elif price > 0.2:  
            
            if battery_soc > 0.2:
                batt_discharge = min(5.0, hvac_power + 1.0)  
                batt_charge = 0.0
        else:  
            if pv_gen > 1.0 and battery_soc < 0.8:
                batt_charge = min(pv_gen - 1.0, 5.0)  
                batt_discharge = 0.0
        
        return np.array([hvac_power, batt_charge, batt_discharge])
    
    env = HEMSEnvironment(test_data)
    opt_layer = OptimizationLayer()
    
    baseline_rewards = []
    baseline_summaries = []
    baseline_dfs = []
    
    for episode in range(len(rl_results['rewards'])):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action_baseline = rule_based_policy(obs)
            
            action = opt_layer.enforce_constraints(obs, action_baseline)
            
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
        
        summary, df = env.get_episode_summary()
        
        baseline_rewards.append(episode_reward)
        baseline_summaries.append(summary)
        baseline_dfs.append(df)
    
    baseline_avg_reward = np.mean(baseline_rewards)
    baseline_avg_cost = np.mean([s['total_energy_cost'] for s in baseline_summaries])
    baseline_avg_comfort = np.mean([s['average_comfort'] for s in baseline_summaries])
    baseline_avg_emissions = np.mean([s['total_carbon_emissions'] for s in baseline_summaries])
    baseline_avg_peak = np.mean([s['peak_demand'] for s in baseline_summaries])
    
    print("\nComparison with Baseline:")
    print(f"{'Metric':<20} {'RL Agent':<12} {'Baseline':<12} {'Improvement (%)':<15}")
    print("-" * 60)
    
    cost_improvement = (baseline_avg_cost - rl_results['avg_cost']) / baseline_avg_cost * 100
    comfort_improvement = (rl_results['avg_comfort'] - baseline_avg_comfort) / baseline_avg_comfort * 100
    emissions_improvement = (baseline_avg_emissions - rl_results['avg_emissions']) / baseline_avg_emissions * 100
    peak_improvement = (baseline_avg_peak - rl_results['avg_peak']) / baseline_avg_peak * 100
    
    print(f"{'Energy Cost':<20} {rl_results['avg_cost']:<12.2f} {baseline_avg_cost:<12.2f} {cost_improvement:<15.2f}")
    print(f"{'Comfort':<20} {rl_results['avg_comfort']:<12.2f} {baseline_avg_comfort:<12.2f} {comfort_improvement:<15.2f}")
    print(f"{'Carbon Emissions':<20} {rl_results['avg_emissions']:<12.2f} {baseline_avg_emissions:<12.2f} {emissions_improvement:<15.2f}")
    print(f"{'Peak Demand':<20} {rl_results['avg_peak']:<12.2f} {baseline_avg_peak:<12.2f} {peak_improvement:<15.2f}")
    
    comparison_df = pd.DataFrame({
        'Metric': ['Energy Cost', 'Comfort', 'Carbon Emissions', 'Peak Demand'],
        'RL Agent': [rl_results['avg_cost'], rl_results['avg_comfort'], rl_results['avg_emissions'], rl_results['avg_peak']],
        'Baseline': [baseline_avg_cost, baseline_avg_comfort, baseline_avg_emissions, baseline_avg_peak],
        'Improvement (%)': [cost_improvement, comfort_improvement, emissions_improvement, peak_improvement]
    })
    
    comparison_df.to_csv(os.path.join(log_dir, 'comparison_results.csv'), index=False)
    
    return {
        'rl': rl_results,
        'baseline': {
            'rewards': baseline_rewards,
            'summaries': baseline_summaries,
            'dfs': baseline_dfs,
            'avg_reward': baseline_avg_reward,
            'avg_cost': baseline_avg_cost,
            'avg_comfort': baseline_avg_comfort,
            'avg_emissions': baseline_avg_emissions,
            'avg_peak': baseline_avg_peak
        },
        'improvements': {
            'cost': cost_improvement,
            'comfort': comfort_improvement,
            'emissions': emissions_improvement,
            'peak': peak_improvement
        }
    }


def create_real_time_controller(agent, opt_layer=None):
    """Create a real-time controller function from a trained agent"""
    if opt_layer is None:
        opt_layer = OptimizationLayer()
    
    def controller(obs):
        """Real-time controller function"""
        action_rl, _ = agent.predict(obs, deterministic=True)
        
        action = opt_layer.enforce_constraints(obs, action_rl)
        
        hvac_power, batt_charge, batt_discharge = action
        
        response = {
            'hvac_power': float(hvac_power),
            'battery_charge': float(batt_charge),
            'battery_discharge': float(batt_discharge),
            'explanation': {
                'hvac': f"Setting HVAC power to {hvac_power:.2f} kW",
                'battery': (
                    f"{'Charging' if batt_charge > 0 else 'Discharging'} battery at "
                    f"{max(batt_charge, batt_discharge):.2f} kW"
                ),
                'reasoning': get_decision_reasoning(obs, action)
            }
        }
        
        return response
    
    def get_decision_reasoning(obs, action):
        """Generate human-readable reasoning for decisions"""
        temp_in, battery_soc, pv_gen, price, hour, carbon = obs
        hvac_power, batt_charge, batt_discharge = action
        
        reasons = []
        
        if hvac_power > 2.0:
            reasons.append("High HVAC usage due to significant temperature deviation")
        elif hvac_power > 0.5:
            reasons.append("Moderate HVAC usage to maintain comfort")
        else:
            reasons.append("Minimal HVAC usage as temperature is within comfort zone")
        
        if batt_charge > 0:
            if price < 0.15:
                reasons.append("Charging battery during low-price period")
            elif pv_gen > batt_charge:
                reasons.append("Charging battery with excess solar generation")
            else:
                reasons.append("Charging battery to prepare for future usage")
        
        if batt_discharge > 0:
            if price > 0.20:
                reasons.append("Discharging battery during high-price period")
            elif pv_gen < 0.5:
                reasons.append("Discharging battery due to low solar availability")
            else:
                reasons.append("Discharging battery to support current load")
        
        return " | ".join(reasons)
    
    return controller 
