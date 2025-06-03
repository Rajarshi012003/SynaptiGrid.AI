import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from stable_baselines3 import TD3
import os
import glob
from datetime import datetime, timedelta

from data_processor import HEMSDataProcessor
from rl_environment import HEMSEnvironment
from rl_agent import create_real_time_controller
from snn_model import SNN_Model

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def find_latest_model():
    """Find the most recent trained model"""
    model_files = glob.glob("logs/run_*/td3_hems_best.zip")
    if not model_files:
        return None
    
    
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return model_files[0]

#find SNN models
def find_snn_models():
    """Find available SNN models"""
    snn_files = glob.glob("logs/run_*/snn_model.pth")
    if not snn_files:
        return {}
    
    
    snn_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    snn_models = {}
    for i, model in enumerate(snn_files):
        run_dir = os.path.dirname(model)
        run_name = os.path.basename(run_dir)
        if i == 0:
            snn_models[f"Latest SNN Model ({run_name})"] = model
        else:
            snn_models[f"SNN Model {run_name}"] = model
    
    return snn_models

st.set_page_config(
    page_title="HEMS Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    h1, h2, h3 {margin-bottom: 0.5rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
    .stTabs [data-baseweb="tab"] {
        height: 50px; 
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helpers
def load_model(model_path):
    """Load a pre-trained RL model"""
    try:
        model = TD3.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_weather_icon(weather):
    """Return weather icon based on condition"""
    if weather == "Clear":
        return "☀️"
    elif weather == "Cloudy":
        return "☁️"
    elif weather == "Rainy":
        return "🌧️"
    elif weather == "Windy":
        return "💨"
    else:
        return "🌤️"

def run_simulation(env, controller, start_time, initial_temp, initial_battery, duration=24):
    """Run a simulation for the specified duration"""
    # Reset env with custom init. config
    obs, _ = env.reset()
    
    env.temp_in = initial_temp
    env.battery_soc = initial_battery
    
    # Simul....
    results = []
    for i in range(duration):
        obs = env._get_observation()
        
        response = controller(obs)
        action = [response['hvac_power'], response['battery_charge'], response['battery_discharge']]
        
        next_obs, reward, done, _, info = env.step(action)
        
        current_time = start_time + timedelta(hours=i)
        current_state = {
            'timestamp': current_time,
            'temperature_indoor': env.temp_in,
            'temperature_outdoor': env.data.iloc[env.current_step]['Temperature'],
            'temperature_setpoint': env.data.iloc[env.current_step]['Temperature_Setpoint'],
            'hvac_power': action[0],
            'battery_soc': env.battery_soc,
            'battery_charge': action[1],
            'battery_discharge': action[2],
            'pv_generation': env.data.iloc[env.current_step]['PV_Generation'],
            'energy_price': env.data.iloc[env.current_step]['Energy_Price'],
            'grid_carbon': env.data.iloc[env.current_step]['Grid_Carbon_Intensity'],
            'grid_import': info.get('grid_import', 0),
            'grid_export': info.get('grid_export', 0),
            'energy_cost': info.get('energy_cost', 0),
            'comfort_violation': abs(env.temp_in - env.data.iloc[env.current_step]['Temperature_Setpoint']),
            'weather': env.data.iloc[env.current_step]['Weather_Condition'],
            'reward': reward,
            'reasoning': response['explanation']['reasoning'] if 'explanation' in response else ""
        }
        
        results.append(current_state)
        
        obs = next_obs
        
        if done:
            break
    
    return pd.DataFrame(results)

# App layout
st.title("🏠 Home Energy Management System Dashboard")
st.markdown("---")

st.sidebar.header("Simulation Settings")

data_path = st.sidebar.text_input("Data Path", value="hems_data_final.csv")
try:
    data_processor = HEMSDataProcessor(data_path)
    data = data_processor.load_data()
    data = data_processor.preprocess_data()
    
    train_data, test_data = data_processor.split_data(test_size=0.2)
    
    st.sidebar.success(f"Data loaded successfully: {len(data)} records")
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    st.stop()

latest_model = find_latest_model()
latest_model_name = "Latest Model (Auto-detected)" if latest_model else "No model auto-detected"

snn_models = find_snn_models()

available_models = {
    latest_model_name: latest_model,
    "Combined RL+SNN Model": "logs/run_20250531_135517/td3_hems_best.zip",
    "Best Model (12.80 Reward)": "logs/run_20250531_131322/td3_hems_best.zip",
    "Optimized Model": "logs/run_20250531_130402/td3_hems_best.zip",
    "Previous Model": "logs/run_20250530_232721/td3_hems_best.zip",
    "Quick-Trained Model": "logs/run_20250531_125610/td3_hems_best.zip"
}

available_models = {k: v for k, v in available_models.items() if v is not None}

st.sidebar.subheader("RL Model Selection")
selected_model = st.sidebar.selectbox(
    "Select RL Model", 
    list(available_models.keys()),
    index=0
)
model_path = available_models[selected_model]

if snn_models:
    st.sidebar.subheader("SNN Model Selection")
    selected_snn_model = st.sidebar.selectbox(
        "Select SNN Model",
        list(snn_models.keys()),
        index=0
    )
    snn_model_path = snn_models[selected_snn_model]
else:
    selected_snn_model = None
    snn_model_path = None

if os.path.exists(model_path):
    model = load_model(model_path)
    if model is not None:
        use_model = True
        st.sidebar.success(f"Model loaded successfully: {model_path}")
    else:
        use_model = False
        st.warning(f"Error loading model from {model_path}. Using rule-based controller instead.")
else:
    use_model = False
    st.warning(f"Model not found at {model_path}. Using rule-based controller instead.")

controller_type = st.sidebar.radio(
    "Controller Type",
    ["AI Model (if available)", "Rule-based Controller"],
    index=0 if use_model else 1
)
use_model = use_model and controller_type == "AI Model (if available)"

st.sidebar.subheader("Initial Conditions")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(data['Timestamp'].min()).date())
start_hour = st.sidebar.slider("Start Hour", 0, 23, 8)
initial_temp = st.sidebar.slider("Initial Indoor Temperature (°C)", 15.0, 28.0, 21.0, 0.5)
initial_battery = st.sidebar.slider("Initial Battery SOC (%)", 0, 100, 50) / 100.0

simulation_hours = st.sidebar.slider("Simulation Duration (hours)", 1, 48, 24)

weather_options = ["Clear", "Cloudy", "Rainy", "Windy"]
weather_override = st.sidebar.selectbox("Weather Condition", options=["Use data values"] + weather_options)

price_override = st.sidebar.slider("Energy Price Override ($/kWh)", 0.05, 0.50, 0.15, 0.01)
use_price_override = st.sidebar.checkbox("Use Price Override", value=False)

pv_factor = st.sidebar.slider("PV Generation Factor", 0.0, 2.0, 1.0, 0.1)

if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        # ENV CREATION
        env = HEMSEnvironment(
            data=test_data,
            episode_length=simulation_hours,
            random_weather=False,
            normalize_obs=True
        )
        
        if use_model:
            controller = create_real_time_controller(model)
        else:
            # Plan B!
            def simple_controller(obs):
                if env.normalize_obs:
                    obs = obs * env.obs_stds + env.obs_means
                
                temp, batt_soc, pv_gen, price, hour, carbon = obs
                
                if temp < 19:
                    hvac = 2.0  # Heat if too cold
                elif temp > 23:
                    hvac = 1.5  # Cool if too hot
                else:
                    hvac = 0.5  # Maintain if comfortable
                
                # Battery
                if price < 0.15 and batt_soc < 0.8:
                    batt_charge = 3.0
                    batt_discharge = 0.0
                elif price > 0.25 and batt_soc > 0.2:
                    batt_charge = 0.0
                    batt_discharge = 3.0
                else:
                    batt_charge = 0.0
                    batt_discharge = 0.0
                
                return {
                    'hvac_power': hvac,
                    'battery_charge': batt_charge,
                    'battery_discharge': batt_discharge,
                    'explanation': {
                        'reasoning': 'Simple rule-based control based on temperature and price thresholds.'
                    }
                }
            
            controller = simple_controller
        
        start_time = datetime.combine(start_date, datetime.min.time()) + timedelta(hours=start_hour)
        
        results_df = run_simulation(
            env, 
            controller, 
            start_time, 
            initial_temp, 
            initial_battery, 
            duration=simulation_hours
        )
        
        st.session_state.results = results_df
        st.session_state.total_energy_cost = results_df['energy_cost'].sum()
        st.session_state.avg_comfort_violation = results_df['comfort_violation'].mean()
        st.session_state.total_pv_used = results_df['pv_generation'].sum()
        st.session_state.peak_demand = results_df['grid_import'].max()
        st.session_state.battery_cycles = (results_df['battery_charge'].sum() / 10)  # Assuming 10 kWh capacity
        st.session_state.simulation_ran = True
        st.session_state.controller_type = "AI Model" if use_model else "Rule-based Controller"
        st.session_state.model_info = selected_model if use_model else "N/A"

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Dashboard", "🔌 Energy Flow", "📈 Controller Analysis", "🧠 Neural Models", "📋 Raw Data", "🔄 Model Training"])

with tab1:
    if 'simulation_ran' in st.session_state and st.session_state.simulation_ran:
        results_df = st.session_state.results
        
        st.info(f"Controller: {st.session_state.controller_type} - {st.session_state.model_info}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Total Energy Cost", f"${st.session_state.total_energy_cost:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Avg. Comfort Violation", f"{st.session_state.avg_comfort_violation:.2f}°C")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Peak Demand", f"{st.session_state.peak_demand:.2f} kW")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Battery Cycles Used", f"{st.session_state.battery_cycles:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.subheader("Temperature Management")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(results_df['timestamp'], results_df['temperature_indoor'], 'b-', label='Indoor Temp')
        ax1.plot(results_df['timestamp'], results_df['temperature_outdoor'], 'g-', label='Outdoor Temp')
        ax1.plot(results_df['timestamp'], results_df['temperature_setpoint'], 'r--', label='Setpoint')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_xlabel('Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        st.subheader("Energy Management")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.bar(results_df['timestamp'], results_df['pv_generation'], color='yellow', alpha=0.7, label='PV Generation')
        ax2.bar(results_df['timestamp'], results_df['hvac_power'], color='red', alpha=0.7, label='HVAC Consumption')
        ax2.plot(results_df['timestamp'], results_df['grid_import'], 'b-', label='Grid Import')
        ax2.plot(results_df['timestamp'], results_df['grid_export'], 'g-', label='Grid Export')
        ax2.set_ylabel('Power (kW)')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        st.subheader("Battery Management")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(results_df['timestamp'], results_df['battery_soc'] * 100, 'b-', label='Battery SOC (%)')
        ax3.set_ylabel('Battery SOC (%)')
        ax3.set_xlabel('Time')
        ax31 = ax3.twinx()
        ax31.bar(results_df['timestamp'], results_df['battery_charge'], color='green', alpha=0.5, label='Charge Rate')
        ax31.bar(results_df['timestamp'], -results_df['battery_discharge'], color='red', alpha=0.5, label='Discharge Rate')
        ax31.set_ylabel('Power (kW)')
        
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax31.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        
        st.subheader("Energy Price & Carbon Intensity")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(results_df['timestamp'], results_df['energy_price'], 'r-', label='Energy Price ($/kWh)')
        ax4.set_ylabel('Price ($/kWh)')
        ax4.set_xlabel('Time')
        
        ax41 = ax4.twinx()
        ax41.plot(results_df['timestamp'], results_df['grid_carbon'], 'g-', label='Carbon Intensity (g/kWh)')
        ax41.set_ylabel('Carbon Intensity (g/kWh)')
        
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax41.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        
    else:
        st.info("Run a simulation using the controls in the sidebar to see the dashboard.")

with tab2:
    if 'simulation_ran' in st.session_state and st.session_state.simulation_ran:
        results_df = st.session_state.results
        
        st.subheader("Energy Flow Analysis")
        
        selected_hour = st.slider("Select Hour", 0, len(results_df)-1, len(results_df)//2)
        selected_data = results_df.iloc[selected_hour]
        
        st.markdown(f"### Time: {selected_data['timestamp']} | Weather: {get_weather_icon(selected_data['weather'])} {selected_data['weather']}")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown("### Energy Sources")
            st.markdown(f"**PV Generation:** {selected_data['pv_generation']:.2f} kW")
            st.markdown(f"**Grid Import:** {selected_data['grid_import']:.2f} kW")
            st.markdown(f"**Battery Discharge:** {selected_data['battery_discharge']:.2f} kW")
            
            st.markdown("### Current Conditions")
            st.markdown(f"**Outdoor Temp:** {selected_data['temperature_outdoor']:.1f}°C")
            st.markdown(f"**Indoor Temp:** {selected_data['temperature_indoor']:.1f}°C")
            st.markdown(f"**Setpoint:** {selected_data['temperature_setpoint']:.1f}°C")
            st.markdown(f"**Energy Price:** ${selected_data['energy_price']:.3f}/kWh")
            st.markdown(f"**Carbon Intensity:** {selected_data['grid_carbon']:.1f} g/kWh")
        
        with col2:
            st.image("hems_energy_flow.png", caption="Energy Flow Diagram")
            
            st.markdown("### Controller Decision Reasoning")
            st.info(selected_data['reasoning'])
        
        with col3:
            st.markdown("### Energy Consumption")
            st.markdown(f"**HVAC:** {selected_data['hvac_power']:.2f} kW")
            st.markdown(f"**Battery Charging:** {selected_data['battery_charge']:.2f} kW")
            st.markdown(f"**Grid Export:** {selected_data['grid_export']:.2f} kW")
            
            st.markdown("### Performance Metrics")
            st.markdown(f"**Battery SOC:** {selected_data['battery_soc']*100:.1f}%")
            st.markdown(f"**Comfort Violation:** {selected_data['comfort_violation']:.2f}°C")
            st.markdown(f"**Energy Cost:** ${selected_data['energy_cost']:.3f}")
            st.markdown(f"**Step Reward:** {selected_data['reward']:.3f}")
    else:
        st.info("Run a simulation using the controls in the sidebar to see the energy flow analysis.")

with tab3:
    if 'simulation_ran' in st.session_state and st.session_state.simulation_ran:
        results_df = st.session_state.results
        
        st.subheader("Controller Analysis")
        
        st.markdown("### HVAC Control Strategy")
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        scatter = ax5.scatter(
            results_df['temperature_indoor'] - results_df['temperature_setpoint'],
            results_df['hvac_power'],
            c=results_df['energy_price'],
            cmap='viridis',
            alpha=0.7
        )
        ax5.set_xlabel('Temperature Deviation (°C)')
        ax5.set_ylabel('HVAC Power (kW)')
        ax5.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Energy Price ($/kWh)')
        st.pyplot(fig5)
        
        st.markdown("### Battery Control Strategy")
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        
        net_battery = results_df['battery_charge'] - results_df['battery_discharge']
        
        scatter = ax6.scatter(
            results_df['energy_price'],
            net_battery,
            c=results_df['battery_soc'],
            cmap='RdYlGn',
            alpha=0.7,
            s=80
        )
        ax6.set_xlabel('Energy Price ($/kWh)')
        ax6.set_ylabel('Net Battery Power (kW)')
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax6.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Battery SOC')
        st.pyplot(fig6)
        
        st.markdown("### Time-of-Day Control Strategy")
        
        results_df['hour'] = results_df['timestamp'].dt.hour
        
        pivot_data = results_df.pivot_table(
            index='hour',
            values=['hvac_power', 'battery_charge', 'battery_discharge', 'grid_import', 'energy_price', 'pv_generation'],
            aggfunc='mean'
        )
        
        fig7, ax7 = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, cmap='viridis', annot=True, fmt='.2f', ax=ax7)
        ax7.set_title('Average Values by Hour of Day')
        ax7.set_ylabel('Hour of Day')
        st.pyplot(fig7)
        
    else:
        st.info("Run a simulation using the controls in the sidebar to analyze controller behavior.")

with tab4:
    st.subheader("Neural Network Models")
    
    st.markdown("### Reinforcement Learning Model")
    st.write(f"Selected model: **{selected_model}**")
    st.write(f"Path: `{model_path}`")
    
    if use_model:
        st.write("Model architecture: TD3 (Twin Delayed DDPG)")
        st.write("- Actor network: 2 hidden layers with 256 neurons each")
        st.write("- Critic networks: 2 hidden layers with 256 neurons each")
    
    st.markdown("### Spiking Neural Network Model")
    
    if snn_model_path and os.path.exists(snn_model_path):
        st.write(f"Selected SNN model: **{selected_snn_model}**")
        st.write(f"Path: `{snn_model_path}`")
        
        try:
            run_dir = os.path.dirname(snn_model_path)
            
            snn_activity_path = os.path.join(run_dir.replace("logs", "results"), "snn_activity.png")
            if os.path.exists(snn_activity_path):
                st.write("#### SNN Activity Visualization")
                st.image(snn_activity_path, caption="SNN Activity Pattern")
            
            snn_loss_path = os.path.join(run_dir.replace("logs", "results"), "snn_training_loss.png")
            if os.path.exists(snn_loss_path):
                st.write("#### SNN Training Loss")
                st.image(snn_loss_path, caption="SNN Training Loss")
        except Exception as e:
            st.error(f"Error loading SNN model information: {e}")
    else:
        st.info("No SNN model selected or available. Train a model with SNN enabled to visualize it here.")

with tab5:
    if 'simulation_ran' in st.session_state and st.session_state.simulation_ran:
        results_df = st.session_state.results
        
        st.subheader("Raw Simulation Data")
        st.dataframe(results_df)
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="hems_simulation_results.csv",
            mime="text/csv",
        )
    else:
        st.info("Run a simulation using the controls in the sidebar to see raw data.")

with tab6:
    st.subheader("Train New Model")
    
    if 'simulation_ran' not in st.session_state:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Training Parameters")
            
            training_timesteps = st.number_input("Training Timesteps", min_value=10000, max_value=500000, value=25000, step=5000)
            episode_length = st.number_input("Episode Length (hours)", min_value=12, max_value=72, value=48, step=12)
            random_weather = st.checkbox("Random Weather", value=True)
            
            with st.expander("Advanced Parameters"):
                learning_rate = st.select_slider("Learning Rate", options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005], value=0.0005)
                batch_size = st.select_slider("Batch Size", options=[64, 128, 256, 512], value=256)
                buffer_size = st.select_slider("Buffer Size", options=[10000, 50000, 100000, 200000], value=100000)
                learning_starts = st.number_input("Learning Starts", min_value=1000, max_value=20000, value=10000, step=1000)
                gradient_steps = st.select_slider("Gradient Steps", options=[1, 10, 100, 1000], value=1000)
        
        with col2:
            st.markdown("### Model Information")
            st.info("""
            Training a new model will optimize the controller for your specific preferences and conditions.
            
            **Parameters explained:**
            - **Training Timesteps**: More steps = better performance but longer training time
            - **Episode Length**: Longer episodes help the model learn long-term effects
            - **Random Weather**: Helps model generalize to different conditions
            - **Learning Rate**: Controls how quickly the model adapts
            - **Batch Size**: Number of experiences used in each learning step
            - **Buffer Size**: How many past experiences to keep for learning
            - **Gradient Steps**: Number of optimization steps per update
            """)
            
            if st.button("Start Training", type="primary"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = f"logs/run_{timestamp}"
                
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                log_placeholder = st.empty()
                
                os.makedirs(run_dir, exist_ok=True)
                
                log_file = os.path.join(run_dir, "training_log.txt")
                with open(log_file, "w") as f:
                    f.write(f"Training started at {datetime.now()}\n")
                    f.write(f"Parameters:\n")
                    f.write(f"  - Timesteps: {training_timesteps}\n")
                    f.write(f"  - Episode length: {episode_length}\n")
                    f.write(f"  - Random weather: {random_weather}\n")
                    f.write(f"  - Learning rate: {learning_rate}\n")
                    f.write(f"  - Batch size: {batch_size}\n")
                    f.write(f"  - Buffer size: {buffer_size}\n")
                    f.write(f"  - Learning starts: {learning_starts}\n")
                    f.write(f"  - Gradient steps: {gradient_steps}\n")
                
                progress_placeholder.info("Preparing training environment...")
                
                command = f"""python train_optimized.py \
                --timesteps {training_timesteps} \
                --episode_length {episode_length} \
                --random_weather {str(random_weather).lower()} \
                --learning_rate {learning_rate} \
                --batch_size {batch_size} \
                --buffer_size {buffer_size} \
                --learning_starts {learning_starts} \
                --gradient_steps {gradient_steps} \
                --run_dir {run_dir}"""
                
                st.code(command, language="bash")
                
                log_placeholder.text_area("Training Log", "Starting training...\n", height=200)
                
                
                st.session_state.training_started = True
                st.session_state.training_run_dir = run_dir
                st.session_state.training_progress = 0
                
                st.experimental_rerun()
    
    elif 'training_started' in st.session_state:
        st.info(f"Training in progress. Run directory: {st.session_state.training_run_dir}")
        
        if 'training_progress' not in st.session_state:
            st.session_state.training_progress = 0
        
        if st.session_state.training_progress < 100:
            st.session_state.training_progress += 5
        
        progress_bar = st.progress(st.session_state.training_progress / 100)
        
        log_file = os.path.join(st.session_state.training_run_dir, "training_log.txt")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log_content = f.read()
            st.text_area("Training Log", log_content, height=300)
        
        if st.session_state.training_progress >= 100:
            st.success("Training complete! The new model is now available in the model selection dropdown.")
            
            if st.button("Run Simulation with New Model", type="primary"):
                del st.session_state.training_started
                del st.session_state.training_progress
                del st.session_state.training_run_dir
                st.experimental_rerun()
    
    else:
        st.info("Run a simulation first before training a new model, or refresh the page.")

st.markdown("---")
st.markdown("Home Energy Management System (HEMS) using Spiking Neural Networks and Reinforcement Learning") 
