# 🏡⚡ Home Energy Management System (HEMS)

A sophisticated controller powered by **Spiking Neural Networks (SNN)** and **Reinforcement Learning (RL)** to optimize home energy usage, comfort, and environmental sustainability.

![HEMS Dashboard](hems_dashboard_screenshot.png)

---

## 🧠 Overview

This intelligent HEMS system:

✅ Optimizes **HVAC** operations
✅ Manages **battery storage** intelligently
✅ Minimizes **energy costs** while maintaining comfort
✅ Reduces **carbon emissions**
✅ Adapts to **weather** & **electricity prices**
✅ Provides an **interactive dashboard** for real-time control

---

## 🏗️ System Architecture

### 🔧 Core Components

| Component           | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| `rl_environment.py` | Simulates home energy with thermal, grid, and battery dynamics |
| `rl_agent.py`       | TD3 agent that learns optimal control policies                 |
| `snn_model.py`      | Spiking Neural Network for energy signal processing            |
| `data_processor.py` | Handles preprocessing, normalization, and feature encoding     |
| `hems_dashboard.py` | **Streamlit** dashboard for visualization and control          |

### 🧬 Neural Architecture

#### 🎯 **Reinforcement Learning (TD3)**

* Actor: `256-256` neurons
* Critic: `256-256` neurons
* Feature extractor with **layer normalization**

#### ⚡ **Spiking Neural Network (SNN)**

* Input: `6 neurons`
* Hidden: `128 LIF neurons`
* Output: `3 neurons`
* Learning: **STDP** (Spike-Timing-Dependent Plasticity)

---

## 🚀 Features

### 🧠 Intelligence & Optimization

* 🎯 **Multi-objective Optimization**: Cost, comfort, carbon footprint
* 🔄 **Adaptive Control**: Dynamic strategy adjustment
* 📈 **Prediction**: Anticipates energy needs & price trends
* 🧩 **Context-Aware Decisions**: Uses weather, time, user settings
* ✅ **Constraint Enforcement**: Ensures valid actions via optimization layer

### 🖥️ User Interface Highlights

* 📊 **Interactive Dashboard**: Live control and insights
* 🌞 **Energy Flow Analysis**: Source & consumption breakdown
* 🌡️ **Temperature Monitoring**: Indoor vs. outdoor
* 🔋 **Battery Visualization**: Charge & discharge cycles
* 🧪 **Controller Analysis**: RL + SNN behavior tracking
* 🔬 **Neural Model Viewer**: SNN spikes and learning visualization

---

## 📈 Models & Performance

| 🧪 Model Type      | 📂 Path                                      | 🎯 Reward | 📝 Description            |
| ------------------ | -------------------------------------------- | --------- | ------------------------- |
| 🔁 RL + SNN Hybrid | `logs/run_20250531_135517/td3_hems_best.zip` | TBD       | Latest hybrid integration |
| 🏆 Best RL Model   | `logs/run_20250531_131322/td3_hems_best.zip` | 12.80     | Best overall performer    |

---

## ⚙️ Installation

```bash
# 📥 Clone the repository
git clone https://github.com/yourusername/hems-research.git
cd hems-research

# 🧪 Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 📦 Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🖥️ Run Dashboard

```bash
./run_dashboard.sh
# or
streamlit run hems_dashboard.py
```

### 🎓 Train a Model

```bash
./train_model.sh
# or
python train_optimized.py --timesteps 50000 --episode_length 48 --snn_epochs 15
```

### 🧪 Test a Model

```bash
python test_model.py --model logs/run_20250531_135517/td3_hems_best.zip
```

---

## 🧰 Key Improvements

### 🏠 Environment

* 🔄 Observation normalization
* 🧠 Smarter reward shaping
* ⚡ Battery model refinement
* 🌡️ Thermal inertia modeling
* 🌍 Realistic environmental variation

### 🤖 RL Agent

* 🧱 Custom feature extractor
* 📉 Learning rate scheduling
* 🧪 Gradient clipping
* 🧠 Ornstein-Uhlenbeck noise for exploration
* 📊 Curricular learning with progressive complexity

### 🔌 SNN Integration

* 🧬 SNN pre-processing of time-series signals
* 🔁 Efficient spike encoding
* 🔧 Tuned hyperparameters for energy domains

---

## 🏆 Results Summary

| ✅ Feature        | 🔍 Outcome                             |
| ---------------- | -------------------------------------- |
| Battery Usage    | Context-sensitive charging/discharging |
| HVAC Control     | Temperature kept within ±0.5°C         |
| Grid Interaction | Reduced peak grid consumption          |
| Solar Use        | Maximized self-consumption             |
| Cost Savings     | 15–25% lower than rule-based control   |

---

## 📂 Project Structure

```bash
.
├── rl_environment.py          # Energy & thermal dynamics
├── rl_agent.py                # TD3 policy agent
├── snn_model.py               # Spiking Neural Network
├── data_processor.py          # Preprocessing logic
├── hems_dashboard.py          # Interactive UI
├── train_optimized.py         # Custom training script
├── train_model.sh             # Quick training launcher
├── run_dashboard.sh           # UI launcher
├── test_model.py              # Evaluation tools
├── hems_data_final.csv        # Dataset used
├── logs/                      # Model logs & checkpoints
└── results/                   # Visualizations & plots
```

---

## 🧭 Future Work

* 🏠 Real-world smart home deployment
* 🌐 Federated learning for multi-home systems
* 🧑‍💼 Preference learning over time
* 🔌 Integration with smart appliances
* 🧠 Explainable AI for control decisions

---

## 📜 License

Licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

This system is built on the **HEMS PLAN REPORT**, which proposes a unified mathematical framework for energy optimization using **SNN** and **RL** techniques.
