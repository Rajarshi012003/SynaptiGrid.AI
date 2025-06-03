#!/usr/bin/env python3
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import argparse

# Avoid Qt errors
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# nonninteractive backend
import matplotlib
matplotlib.use('Agg')

from data_processor import HEMSDataProcessor
from rl_environment import HEMSEnvironment
from rl_agent import train_rl_agent, evaluate_rl_agent
from snn_model import SNN_Model, train_snn_model, test_snn_model, visualize_snn_activity

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a HEMS RL model with optimized parameters')
    
    parser.add_argument('--timesteps', type=int, default=25000, help='Number of timesteps to train for')
    parser.add_argument('--episode_length', type=int, default=48, help='Length of each episode in hours')
    parser.add_argument('--random_weather', type=lambda x: x.lower() == 'true', default=True, help='Whether to randomize weather')
    
    parser.add_argument('--skip_snn', action='store_true', help='Skip SNN training')
    parser.add_argument('--snn_epochs', type=int, default=10, help='Number of epochs for SNN training')
    parser.add_argument('--snn_lr', type=float, default=1e-3, help='Learning rate for SNN training')
    
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_starts', type=int, default=10000, help='Learning starts')
    parser.add_argument('--gradient_steps', type=int, default=1000, help='Gradient steps')
    parser.add_argument('--train_freq', type=int, default=1000, help='Training frequency')
    
    # Output 
    parser.add_argument('--run_dir', type=str, default=None, help='Run directory (if None, will be created)')
    
    return parser.parse_args()

np.random.seed(42)
torch.manual_seed(42)

def setup_dirs(run_dir=None):
    """Create necessary directories for outputs"""
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        run_dir = f"logs/run_{timestamp}"
        model_dir = f"models/run_{timestamp}"
        results_dir = f"results/run_{timestamp}"
    else:
        model_dir = run_dir.replace("logs", "models")
        results_dir = run_dir.replace("logs", "results")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return run_dir, model_dir, results_dir

def main():
    """Main function to run optimized training"""
    args = parse_args()
    
    print("Starting Optimized HEMS Training...")
    print(f"Training configuration:")
    print(f"  - Timesteps: {args.timesteps}")
    print(f"  - Episode length: {args.episode_length}")
    print(f"  - Random weather: {args.random_weather}")
    print(f"  - Skip SNN: {args.skip_snn}")
    if not args.skip_snn:
        print(f"  - SNN epochs: {args.snn_epochs}")
        print(f"  - SNN learning rate: {args.snn_lr}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Buffer size: {args.buffer_size}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning starts: {args.learning_starts}")
    print(f"  - Gradient steps: {args.gradient_steps}")
    print(f"  - Train frequency: {args.train_freq}")
    
    run_dir, model_dir, results_dir = setup_dirs(args.run_dir)
    
    with open(os.path.join(run_dir, 'run_config.txt'), 'w') as f:
        f.write(f"skip_snn: {args.skip_snn}\n")
        f.write(f"snn_epochs: {args.snn_epochs}\n")
        f.write(f"snn_lr: {args.snn_lr}\n")
        f.write(f"timesteps: {args.timesteps}\n")
        f.write(f"episode_length: {args.episode_length}\n")
        f.write(f"random_weather: {args.random_weather}\n")
        f.write(f"learning_rate: {args.learning_rate}\n")
        f.write(f"buffer_size: {args.buffer_size}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"learning_starts: {args.learning_starts}\n")
        f.write(f"gradient_steps: {args.gradient_steps}\n")
        f.write(f"train_freq: {args.train_freq}\n")
    
    print("\nPhase 1: Data Preparation")
    data_processor = HEMSDataProcessor("hems_data_final.csv")
    data_processor.load_data()
    data_processor.preprocess_data()
    data_processor.normalize_data()
    train_data, test_data = data_processor.split_data(test_size=0.2)
    
    if not args.skip_snn:
        print("\nPhase 2: SNN Training")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        train_loader, test_loader, input_size = data_processor.prepare_snn_input(batch_size=64)
        
        snn_model = SNN_Model(input_size)
        train_losses = train_snn_model(snn_model, train_loader, num_epochs=args.snn_epochs, 
                                      lr=args.snn_lr, device=device)
        
        test_loss = test_snn_model(snn_model, test_loader, device=device)
        
        # Save SNN model
        torch.save(snn_model.state_dict(), os.path.join(run_dir, "snn_model.pth"))
        
        # SNN activity
        sample_input = next(iter(test_loader))[0][0]  # Get first input from test set
        visualize_snn_activity(snn_model, sample_input, device=device)
        plt.savefig(os.path.join(results_dir, "snn_activity.png"))
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses)
        plt.title(f'SNN Training Loss ({args.snn_epochs} epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, "snn_training_loss.png"))
        plt.close()
        
        print(f"SNN training completed with test loss: {test_loss:.6f}")
    
    print("\nPhase 3: Setting up RL Environment with Optimized Parameters")
    
    train_env = HEMSEnvironment(
        train_data, 
        episode_length=args.episode_length,
        random_weather=args.random_weather,
        normalize_obs=True
    )
    
    eval_env = HEMSEnvironment(
        test_data, 
        episode_length=args.episode_length,
        random_weather=False,  
        normalize_obs=True
    )
    
    print("\nPhase 4: RL Training with Optimized Parameters")
    
    print(f"Training for {args.timesteps} timesteps...")
    
    start_time = datetime.now()
    
    rl_model = train_rl_agent(
        train_env, 
        eval_env, 
        total_timesteps=args.timesteps,
        log_dir=run_dir,
        #  TD3 hyperparameters
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        learning_starts=args.learning_starts
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Training completed in {duration}")
    
    with open(os.path.join(run_dir, 'training_metadata.txt'), 'w') as f:
        f.write(f"Training started: {start_time}\n")
        f.write(f"Training completed: {end_time}\n")
        f.write(f"Duration: {duration}\n")
        f.write(f"Total timesteps: {args.timesteps}\n")
        f.write(f"Environment parameters:\n")
        f.write(f"  - Episode length: {args.episode_length}\n")
        f.write(f"  - Random weather: {args.random_weather}\n")
        f.write(f"  - Observation normalization: True\n")
        f.write(f"Learning parameters:\n")
        f.write(f"  - Learning rate: {args.learning_rate}\n")
        f.write(f"  - Buffer size: {args.buffer_size}\n")
        f.write(f"  - Batch size: {args.batch_size}\n")
        f.write(f"  - Gradient steps: {args.gradient_steps}\n")
        f.write(f"  - Train frequency: {args.train_freq}\n")
    
    final_model_path = os.path.join(run_dir, "td3_hems_final.zip")
    rl_model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    print("\nPhase 5: Comprehensive Evaluation")
    
    print("Evaluating on test scenarios...")
    results = evaluate_rl_agent(rl_model, eval_env, n_episodes=5)
    
    evaluation_df = pd.DataFrame({
        'Episode': range(1, len(results['rewards']) + 1),
        'Reward': results['rewards'],
        'Energy_Cost': [s['total_energy_cost'] for s in results['summaries']],
        'Comfort': [s['average_comfort'] for s in results['summaries']],
        'Emissions': [s['total_carbon_emissions'] for s in results['summaries']],
        'Peak_Demand': [s['peak_demand'] for s in results['summaries']]
    })
    evaluation_df.to_csv(os.path.join(results_dir, 'evaluation_results.csv'), index=False)
    
    avg_reward = np.mean(results['rewards'])
    avg_cost = np.mean([s['total_energy_cost'] for s in results['summaries']])
    avg_comfort = np.mean([s['average_comfort'] for s in results['summaries']])
    avg_emissions = np.mean([s['total_carbon_emissions'] for s in results['summaries']])
    avg_peak = np.mean([s['peak_demand'] for s in results['summaries']])
    
    print("\nEvaluation Summary:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average energy cost: {avg_cost:.2f}")
    print(f"Average comfort: {avg_comfort:.2f}")
    print(f"Average carbon emissions: {avg_emissions:.2f}")
    print(f"Average peak demand: {avg_peak:.2f}")
    
    with open(os.path.join(results_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write("Evaluation Summary:\n")
        f.write(f"Average reward: {avg_reward:.2f}\n")
        f.write(f"Average energy cost: {avg_cost:.2f}\n")
        f.write(f"Average comfort: {avg_comfort:.2f}\n")
        f.write(f"Average carbon emissions: {avg_emissions:.2f}\n")
        f.write(f"Average peak demand: {avg_peak:.2f}\n")
    
    print(f"\nResults saved to {results_dir}")
    print("\nOptimized HEMS Training Complete!")
    
    best_model_path = os.path.join(run_dir, "td3_hems_best.zip")
    print(f"Best model saved at: {best_model_path}")
    
    return best_model_path

if __name__ == "__main__":
    main() 
