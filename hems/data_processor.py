"""
data preprocessing, normalization, and preparation 
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class HEMSDataProcessor:
    """Class for data preprocessing and preparation"""
    
    def __init__(self, data_path):
        """Initialize with the path to the dataset"""
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.test_data = None
        self.scaler = None
        self.feature_names = None
        
    def load_data(self):
        """Load the dataset"""
        self.raw_data = pd.read_csv(self.data_path)
        
        self.raw_data['Timestamp'] = pd.to_datetime(self.raw_data['Timestamp'])
        print(f"Data loaded: {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
        return self.raw_data
    
    def preprocess_data(self):
        print("Missing values in each column:")
        print(self.raw_data.isnull().sum())
        
       
        self.processed_data = self.raw_data.copy()
        self.processed_data['Battery_Charge'].fillna(0, inplace=True)
        self.processed_data['Battery_Discharge'].fillna(0, inplace=True)
        
        # Handle Net_Consumption 
        if self.processed_data['Net_Consumption'].isnull().sum() > 0:
            self.processed_data['Net_Consumption'] = (
                self.processed_data['Total_Consumption'] - 
                self.processed_data['PV_Generation']
            )
        
        
        # time features
        self.processed_data['Hour'] = self.processed_data['Timestamp'].dt.hour
        self.processed_data['Day'] = self.processed_data['Timestamp'].dt.day
        self.processed_data['Month'] = self.processed_data['Timestamp'].dt.month
        self.processed_data['DayOfWeek'] = self.processed_data['Timestamp'].dt.dayofweek
        
        # energy consumption features
        self.processed_data['Shiftable_Load'] = (
            self.processed_data['Dishwasher'] + 
            self.processed_data['Washing_Machine'] + 
            self.processed_data['Dryer']
        )
        
        self.processed_data['Base_Load'] = (
            self.processed_data['Refrigerator'] + 
            self.processed_data['Lighting']
        )
        
        # power balance
        self.processed_data['Power_Balance'] = (
            self.processed_data['Total_Consumption'] - 
            self.processed_data['PV_Generation'] + 
            self.processed_data['Battery_Discharge'] - 
            self.processed_data['Battery_Charge']
        )
        
        #  (indoor-outdoor) temps
        self.processed_data['Temp_Diff'] = self.processed_data['Temperature_Setpoint'] - self.processed_data['Temperature']
        
        # Weather encoded
        weather_mapping = {'Clear': 0, 'Cloudy': 1, 'Rainy': 2, 'Windy': 3}
        self.processed_data['Weather_Encoded'] = self.processed_data['Weather_Condition'].map(weather_mapping)
        
        print(f"Preprocessing complete: {len(self.processed_data)} rows and {len(self.processed_data.columns)} columns")
        return self.processed_data
    
    def normalize_data(self):
        features = [
            'Temperature', 'Energy_Price', 'HVAC', 'Water_Heater', 
            'Total_Consumption', 'PV_Generation', 'Grid_Carbon_Intensity',
            'Battery_SOC', 'Temperature_Deviation'
        ]
        
        self.scaler = StandardScaler()
        self.processed_data[features] = self.scaler.fit_transform(self.processed_data[features])
        
        print(f"Data normalized for features: {features}")
        return self.processed_data
    
    def split_data(self, test_size=0.2):
        split_idx = int(len(self.processed_data) * (1 - test_size))
        self.train_data = self.processed_data.iloc[:split_idx]
        self.test_data = self.processed_data.iloc[split_idx:]
        
        print(f"Data split: {len(self.train_data)} training samples, {len(self.test_data)} testing samples")
        return self.train_data, self.test_data
    
    def prepare_snn_input(self, batch_size=32):
        """Prepare data for SNN input using spike encoding"""
        # snn features
        snn_features = [
            'Temperature', 'Energy_Price', 'PV_Generation', 
            'Battery_SOC', 'Grid_Carbon_Intensity', 'Hour',
            'Weather_Encoded', 'Temperature_Deviation'
        ]
        self.feature_names = snn_features
        
        
        target_features = [
            'HVAC', 'Battery_Charge', 'Battery_Discharge'
        ]
        
        
        X_train = torch.tensor(self.train_data[snn_features].values, dtype=torch.float32)
        y_train = torch.tensor(self.train_data[target_features].values, dtype=torch.float32)
        X_test = torch.tensor(self.test_data[snn_features].values, dtype=torch.float32)
        y_test = torch.tensor(self.test_data[target_features].values, dtype=torch.float32)
        
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"SNN input prepared: {X_train.shape[1]} features")
        return train_loader, test_loader, X_train.shape[1]
    
    def prepare_rl_data(self):
        # prepare for rl env
        return self.train_data, self.test_data
    
    def encode_spikes(self, data, threshold=0.5, n_steps=100):
        """Encode continuous values to spike trains using rate coding"""
        # Normalize -> [0, 1]
        normalized_data = (data - data.min()) / (data.max() - data.min())
        
        # Create spike trains
        spike_trains = torch.zeros(n_steps, len(data))
        
        for t in range(n_steps):
            # Generate spikes with probability proportional to input value
            spike_trains[t] = torch.bernoulli(normalized_data)
        
        return spike_trains
    
    def get_feature_stats(self):
        """Return statistics for each feature"""
        stats = {}
        for feature in self.processed_data.columns:
            if feature != 'Timestamp' and feature != 'Weather_Condition':
                stats[feature] = {
                    'mean': self.processed_data[feature].mean(),
                    'std': self.processed_data[feature].std(),
                    'min': self.processed_data[feature].min(),
                    'max': self.processed_data[feature].max()
                }
        return stats
    
    def analyze_data_patterns(self):
        """Analyze patterns in the data"""
        # Time-based patterns
        hourly_consumption = self.processed_data.groupby('Hour')['Total_Consumption'].mean()
        hourly_pv = self.processed_data.groupby('Hour')['PV_Generation'].mean()
        hourly_price = self.processed_data.groupby('Hour')['Energy_Price'].mean()
        
        # Temperature vs. HVAC correlation
        temp_hvac_corr = self.processed_data[['Temperature', 'HVAC']].corr().iloc[0, 1]
        
        # Battery usage patterns
        battery_usage = self.processed_data.groupby('Hour')[['Battery_Charge', 'Battery_Discharge']].mean()
        
        results = {
            'hourly_consumption': hourly_consumption,
            'hourly_pv': hourly_pv,
            'hourly_price': hourly_price,
            'temp_hvac_corr': temp_hvac_corr,
            'battery_usage': battery_usage
        }
        
        return results 
