"""
Reinforcement Learning Environment for HEMS

This module implements a custom Gym environment for HEMS control using TD3 Reinforcement Learning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

# Constants
BATTERY_CAPACITY = 10.0  # kWh
BATTERY_POWER = 5.0      # kW
MAX_TEMPERATURE = 30.0   # °C
MIN_TEMPERATURE = 10.0   # °C
THERMAL_RESISTANCE = 2.0  # °C/kW
THERMAL_CAPACITANCE = 2.0  # kWh/°C
HVAC_EFFICIENCY = 3.0    # COP

# Updated reward weights with better balance
COMFORT_WEIGHT = 1.0      # α weight for comfort in objective function (reduced from 2.0)
CARBON_WEIGHT = 0.5       # β weight for carbon emissions in objective function (reduced from 1.0)
COST_WEIGHT = 1.5         # δ weight for energy cost (reduced from 3.0)
PEAK_PENALTY = 0.8        # γ weight for peak demand penalty (reduced from 1.5)
BATTERY_USAGE_REWARD = 0.3  # κ weight for proper battery usage (reduced from 0.5)
REWARD_SCALE = 0.1        # Overall reward scaling factor to prevent exploding gradients

class HEMSEnvironment(gym.Env):
    """Custom Gym environment for HEMS control using TD3 Reinforcement Learning"""
    
    def __init__(self, data, episode_length=24, random_weather=True, normalize_obs=True):
        super().__init__()
        
        # Store the data and initialize state
        self.data = data
        self.current_step = 0
        self.episode_length = episode_length
        self.total_steps = len(data)
        self.episode_start_idx = 0
        self.random_weather = random_weather
        self.normalize_obs = normalize_obs
        
        # Define action and observation spaces
        # Actions: HVAC power, Battery charge rate, Battery discharge rate
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([3.0, BATTERY_POWER, BATTERY_POWER]),
            dtype=np.float32
        )
        
        # Observations: [Temp, Battery SOC, PV Generation, Price, Time, Carbon Intensity]
        self.observation_space = spaces.Box(
            low=np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([40.0, 1.0, 10.0, 1.0, 23.0, 500.0]),
            dtype=np.float32
        )
        
        # Calculate observation normalization parameters
        if normalize_obs:
            self._calculate_norm_params()
        
        # Internal state
        self.temp_in = 21.0  # Initial indoor temperature
        self.battery_soc = 0.5  # Initial battery state of charge
        self.history = []  # For storing episode data
        self.peak_demand_history = []  # For tracking peak demand
        
    def _calculate_norm_params(self):
        """Calculate normalization parameters for observations"""
        # Extract all relevant columns for normalization
        temp = self.data['Temperature'].values
        pv_gen = self.data['PV_Generation'].values
        price = self.data['Energy_Price'].values
        time = np.array([i % 24 for i in range(len(self.data))])  # Hours 0-23
        carbon = self.data['Grid_Carbon_Intensity'].values
        
        # Calculate means and stds
        self.obs_means = np.array([
            np.mean(temp),  # Temperature
            0.5,  # Battery SOC (always 0-1, mean is 0.5)
            np.mean(pv_gen),  # PV generation
            np.mean(price),  # Energy price
            11.5,  # Hour (mean of 0-23)
            np.mean(carbon)  # Carbon intensity
        ])
        
        self.obs_stds = np.array([
            np.std(temp) if np.std(temp) > 0 else 1.0,  # Temperature
            0.3,  # Battery SOC (std of uniform 0-1 is ~0.3)
            np.std(pv_gen) if np.std(pv_gen) > 0 else 1.0,  # PV generation
            np.std(price) if np.std(price) > 0 else 1.0,  # Energy price
            6.9,  # Hour (std of uniform 0-23)
            np.std(carbon) if np.std(carbon) > 0 else 1.0  # Carbon intensity
        ])
        
    def _normalize_observation(self, obs):
        """Normalize observation using mean and std"""
        return (obs - self.obs_means) / self.obs_stds
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Randomly choose a starting point in the data if we have enough data
        if self.total_steps > self.episode_length:
            self.episode_start_idx = self.np_random.integers(0, self.total_steps - self.episode_length)
        else:
            self.episode_start_idx = 0
            
        self.current_step = self.episode_start_idx
        
        # Randomize initial conditions for better generalization
        self.temp_in = self.np_random.uniform(18.0, 24.0)  # Random initial temperature
        self.battery_soc = self.np_random.uniform(0.2, 0.8)  # Random initial battery SOC
        self.history = []
        self.peak_demand_history = []
        
        # Get initial observation
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """Take action in the environment"""
        # Unpack action: [HVAC power, Battery charge, Battery discharge]
        hvac_power, batt_charge, batt_discharge = action
        
        # Get current state from data
        row = self.data.iloc[self.current_step]
        outdoor_temp = row['Temperature']
        pv_gen = row['PV_Generation']
        price = row['Energy_Price']
        carbon = row['Grid_Carbon_Intensity']
        
        # Apply random weather variations if enabled
        if self.random_weather:
            outdoor_temp += self.np_random.normal(0, 1.0)  # Add noise to outdoor temperature
            pv_gen *= (1.0 + self.np_random.normal(0, 0.1))  # Add noise to PV generation
            pv_gen = max(0, pv_gen)  # Ensure non-negative
        
        # Thermal dynamics simulation (improved)
        # Add thermal inertia factor to make temperature changes more realistic
        heat_flow = (outdoor_temp - self.temp_in) / THERMAL_RESISTANCE
        hvac_heat = HVAC_EFFICIENCY * hvac_power  # Convert power to heat flow
        thermal_inertia = self.np_random.uniform(0.8, 1.2)  # Random thermal inertia factor
        self.temp_in += thermal_inertia * (heat_flow + hvac_heat) / THERMAL_CAPACITANCE
        self.temp_in = np.clip(self.temp_in, MIN_TEMPERATURE, MAX_TEMPERATURE)
        
        # Battery dynamics with improved logic
        # Ensure we don't charge and discharge simultaneously with more sophisticated check
        if batt_charge > 0.1 and batt_discharge > 0.1:
            # If both are significant, keep the larger one and zero out the smaller one
            if batt_charge >= batt_discharge:
                batt_discharge = 0
            else:
                batt_charge = 0
        
        # Calculate new battery state with efficiency and degradation factors
        charge_efficiency = 0.92  # 92% charging efficiency
        discharge_efficiency = 0.94  # 94% discharge efficiency
        
        batt_charge_effective = min(batt_charge, BATTERY_POWER)
        batt_discharge_effective = min(batt_discharge, BATTERY_POWER)
        
        # Limit charge/discharge based on current SOC with improved constraints
        max_charge = min(
            batt_charge_effective,
            (1.0 - self.battery_soc) * BATTERY_CAPACITY / charge_efficiency
        )
        
        max_discharge = min(
            batt_discharge_effective,
            self.battery_soc * BATTERY_CAPACITY * discharge_efficiency
        )
        
        # Update battery state of charge with efficiency factors
        self.battery_soc += (charge_efficiency * max_charge - max_discharge / discharge_efficiency) / BATTERY_CAPACITY
        self.battery_soc = max(0.0, min(1.0, self.battery_soc))  # Clip to [0,1]
        
        # Calculate power balance for all appliances
        total_consumption = hvac_power + row['Refrigerator'] + row['Water_Heater'] + \
                           row['Dishwasher'] + row['Washing_Machine'] + row['Dryer'] + \
                           row['EV_Charger'] + row['Lighting']
                           
        # Calculate grid import/export with improved logic for realistic power flow
        grid_import = max(0, total_consumption + max_charge - pv_gen - max_discharge)
        grid_export = max(0, pv_gen + max_discharge - total_consumption - max_charge)
        
        # Keep track of peak demand for more effective peak shaving
        self.peak_demand_history.append(grid_import)
        current_peak = max(self.peak_demand_history)
        
        # Calculate costs and emissions with improved formulation
        energy_cost = price * grid_import
        carbon_emissions = carbon * grid_import / 1000  # Convert to tons CO2
        
        # Calculate comfort deviation (how far from setpoint)
        temp_setpoint = row['Temperature_Setpoint']
        temp_dev = abs(self.temp_in - temp_setpoint)
        
        # Improved comfort function with asymmetric penalty for being too cold vs too hot
        if self.temp_in < temp_setpoint:
            # Being too cold is worse than being too hot
            comfort_utility = 1 / (1 + np.exp(1.5 * (temp_dev - 1.5)))
        else:
            # Being too hot
            comfort_utility = 1 / (1 + np.exp(temp_dev - 2))
        
        # Calculate reward components with improved weights
        cost_component = -COST_WEIGHT * energy_cost
        comfort_component = COMFORT_WEIGHT * comfort_utility
        carbon_component = -CARBON_WEIGHT * carbon_emissions
        
        # Progressive peak penalty (higher penalty for higher peaks)
        if grid_import > 7.0:
            peak_factor = 1.0 + 0.5 * (grid_import - 7.0)  # Increase penalty for high peaks
        else:
            peak_factor = 1.0
        peak_component = -PEAK_PENALTY * peak_factor * max(0, grid_import - 6.0)  # Penalize demand over 6kW
        
        # Battery usage reward with improved shaping
        battery_component = 0
        
        # Reward for appropriate battery usage
        if price > 0.20:  # High price threshold
            # Reward discharging during high price (more if battery SOC is high)
            discharge_factor = 1.0 + self.battery_soc  # Scale by SOC (1.0-2.0)
            battery_component += BATTERY_USAGE_REWARD * max_discharge * discharge_factor
        elif price < 0.10:  # Low price threshold
            # Reward charging during low price (more if battery SOC is low)
            charge_factor = 2.0 - self.battery_soc  # Scale by (1-SOC) (1.0-2.0)
            battery_component += BATTERY_USAGE_REWARD * max_charge * charge_factor
        
        # Additional reward for utilizing PV generation for charging
        if pv_gen > 1.0:
            pv_utilization = min(max_charge, pv_gen) / pv_gen  # Ratio of PV used for charging
            battery_component += BATTERY_USAGE_REWARD * pv_utilization * 0.5
        
        # Reward for maintaining battery SOC in a healthy range (0.2-0.8)
        if self.battery_soc < 0.2:
            # Penalize very low SOC
            battery_component -= BATTERY_USAGE_REWARD * (0.2 - self.battery_soc) * 2.0
        elif self.battery_soc > 0.8:
            # Mild penalty for very high SOC (less severe than low SOC)
            battery_component -= BATTERY_USAGE_REWARD * (self.battery_soc - 0.8)
        
        # Calculate raw reward
        raw_reward = cost_component + comfort_component + carbon_component + peak_component + battery_component
        
        # Apply reward scaling to prevent exploding gradients
        reward = raw_reward * REWARD_SCALE
        
        # Store state information for analysis
        info = {
            'temperature': self.temp_in,
            'battery_soc': self.battery_soc,
            'energy_cost': energy_cost,
            'carbon_emissions': carbon_emissions,
            'comfort_utility': comfort_utility,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'hvac_power': hvac_power,
            'batt_charge': max_charge,
            'batt_discharge': max_discharge,
            'pv_generation': pv_gen,
            'outdoor_temp': outdoor_temp,
            'reward_components': {
                'cost': cost_component,
                'comfort': comfort_component,
                'carbon': carbon_component,
                'peak': peak_component,
                'battery': battery_component
            }
        }
        
        # Save to history
        self.history.append(info)
        
        # Advance to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step - self.episode_start_idx) >= self.episode_length
        truncated = False
        
        # Get next observation
        obs = self._get_observation()
        
        return obs, reward, done, truncated, info
    
    def _get_observation(self):
        """Get current observation from environment state"""
        # Make sure we haven't gone past the data bounds
        current_idx = min(self.current_step, len(self.data) - 1)
        row = self.data.iloc[current_idx]
        
        # Get hour of day from timestamp
        hour = float(row.get('Hour', current_idx % 24))  # Use Hour column if available, otherwise calculate
        
        # Create observation vector
        obs = np.array([
            self.temp_in,  # Current indoor temperature
            self.battery_soc,  # Battery state of charge
            row['PV_Generation'],  # PV generation forecast
            row['Energy_Price'],  # Energy price
            hour,  # Hour of day
            row['Grid_Carbon_Intensity']  # Carbon intensity
        ], dtype=np.float32)
        
        # Apply normalization if enabled
        if self.normalize_obs:
            obs = self._normalize_observation(obs)
        
        return obs
    
    def get_episode_summary(self):
        """Return a summary of the episode"""
        if not self.history:
            return None
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.history)
        
        # Calculate summary statistics
        summary = {
            'total_energy_cost': df['energy_cost'].sum(),
            'average_comfort': df['comfort_utility'].mean(),
            'total_carbon_emissions': df['carbon_emissions'].sum(),
            'peak_demand': df['grid_import'].max(),
            'average_temperature': df['temperature'].mean(),
            'final_battery_soc': df['battery_soc'].iloc[-1],
            'total_pv_generation': df['pv_generation'].sum(),
            'total_grid_import': df['grid_import'].sum(),
            'total_grid_export': df['grid_export'].sum(),
            'self_consumption_ratio': 1.0 - (df['grid_export'].sum() / df['pv_generation'].sum()) if df['pv_generation'].sum() > 0 else 0
        }
        
        return summary, df


class OptimizationLayer:
    """Safety layer to enforce constraints using quadratic programming"""
    
    def __init__(self):
        """Initialize the optimization layer"""
        self.battery_capacity = BATTERY_CAPACITY
        self.battery_power = BATTERY_POWER
        self.charge_efficiency = 0.92
        self.discharge_efficiency = 0.94
        
    def enforce_constraints(self, state, action_rl):
        """Project RL actions onto feasible set"""
        try:
            import cvxpy as cp
            
            # Unpack state and action
            temp_in, battery_soc, pv_gen, price, hour, carbon = state
            hvac_power_rl, batt_charge_rl, batt_discharge_rl = action_rl
            
            # Denormalize state if needed (assuming normalized state)
            if abs(battery_soc) <= 1.0 and abs(temp_in) <= 5.0:  # Heuristic check for normalized values
                # These are reasonable bounds for normalized values
                battery_soc = min(max(battery_soc, 0.0), 1.0)  # Ensure valid SOC range
            
            # Define optimization variables
            hvac_power = cp.Variable(1, nonneg=True)
            batt_charge = cp.Variable(1, nonneg=True)
            batt_discharge = cp.Variable(1, nonneg=True)
            
            # Define objective: minimize distance to RL action
            objective = cp.Minimize(
                cp.sum_squares(hvac_power - hvac_power_rl) + 
                cp.sum_squares(batt_charge - batt_charge_rl) + 
                cp.sum_squares(batt_discharge - batt_discharge_rl)
            )
            
            # Define constraints with improved numerical stability
            # Using proper multiplication operators to avoid CVXPY warnings
            constraints = [
                hvac_power <= 3.0,
                batt_charge <= self.battery_power,
                batt_discharge <= self.battery_power,
                cp.multiply(batt_charge, batt_discharge) <= 1e-4,  # Prevent simultaneous charge/discharge using elementwise multiplication
                batt_charge <= cp.multiply(1.0 - battery_soc, self.battery_capacity / self.charge_efficiency),  # Charge limit
                batt_discharge <= cp.multiply(battery_soc, self.battery_capacity * self.discharge_efficiency)  # Discharge limit
            ]
            
            # Solve problem
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4)
                
                # Extract solution
                if prob.status == "optimal" or prob.status == "optimal_inaccurate":
                    return np.array([
                        float(hvac_power.value), 
                        float(batt_charge.value), 
                        float(batt_discharge.value)
                    ])
                else:
                    # Fall back to simple constraints if optimization fails
                    return self._apply_simple_constraints(action_rl, battery_soc)
            except Exception as e:
                # Fall back to simple constraints if solver fails
                return self._apply_simple_constraints(action_rl, battery_soc)
                
        except Exception as e:
            # Fall back to simple constraints if there's an error
            return self._apply_simple_constraints(action_rl, battery_soc)
    
    def _apply_simple_constraints(self, action_rl, battery_soc):
        """Apply simple constraints without optimization"""
        hvac_power, batt_charge, batt_discharge = action_rl
        
        # Apply constraints
        hvac_power = max(0.0, min(3.0, hvac_power))
        batt_charge = max(0.0, min(self.battery_power, batt_charge))
        batt_discharge = max(0.0, min(self.battery_power, batt_discharge))
        
        # Ensure valid SOC range
        battery_soc = min(max(battery_soc, 0.0), 1.0)
        
        # Prevent simultaneous charge/discharge
        if batt_charge > 0.1 and batt_discharge > 0.1:
            if batt_charge >= batt_discharge:
                batt_discharge = 0.0
            else:
                batt_charge = 0.0
        
        # Limit charge/discharge based on SOC
        batt_charge = min(batt_charge, (1.0 - battery_soc) * self.battery_capacity / self.charge_efficiency)
        batt_discharge = min(batt_discharge, battery_soc * self.battery_capacity * self.discharge_efficiency)
        
        return np.array([hvac_power, batt_charge, batt_discharge])


def visualize_episode(env, episode_df=None):
    """Visualize episode results"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    
    # Skip if visualization is disabled or no data
    if episode_df is None or len(episode_df) == 0:
        return
    
    # Create time axis
    start_time = datetime(2023, 1, 1, 0, 0)
    times = [start_time + timedelta(hours=i) for i in range(len(episode_df))]
    
    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot temperature
    axs[0].plot(times, episode_df['temperature'], 'b-', label='Indoor Temp')
    axs[0].plot(times, episode_df['outdoor_temp'], 'r--', label='Outdoor Temp')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].set_title('Temperature and HVAC Power')
    axs[0].legend(loc='upper left')
    
    # Add HVAC power on second y-axis
    ax_hvac = axs[0].twinx()
    ax_hvac.plot(times, episode_df['hvac_power'], 'g-', label='HVAC Power')
    ax_hvac.set_ylabel('HVAC Power (kW)')
    ax_hvac.legend(loc='upper right')
    
    # Plot energy flows
    axs[1].plot(times, episode_df['grid_import'], 'r-', label='Grid Import')
    axs[1].plot(times, episode_df['grid_export'], 'g-', label='Grid Export')
    axs[1].plot(times, episode_df['pv_generation'], 'y-', label='PV Generation')
    axs[1].set_ylabel('Power (kW)')
    axs[1].set_title('Energy Flows')
    axs[1].legend()
    
    # Plot battery state
    axs[2].plot(times, episode_df['battery_soc'] * 100, 'b-', label='Battery SOC (%)')
    axs[2].set_ylabel('Battery SOC (%)')
    axs[2].set_ylim(0, 100)
    axs[2].set_title('Battery State and Power')
    axs[2].legend(loc='upper left')
    
    # Add battery power on second y-axis
    ax_batt = axs[2].twinx()
    ax_batt.plot(times, episode_df['batt_charge'], 'g-', label='Charge Power')
    ax_batt.plot(times, episode_df['batt_discharge'], 'r-', label='Discharge Power')
    ax_batt.set_ylabel('Battery Power (kW)')
    ax_batt.legend(loc='upper right')
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show() 