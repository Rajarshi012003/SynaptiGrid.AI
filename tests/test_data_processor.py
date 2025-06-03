"""
Just a test for the DATA_PREPROCESSOR!!
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from data_processor import HEMSDataProcessor

def test_data_processor():
    """Test the data processor functionality"""
    print("Testing HEMSDataProcessor...")
    
    data_path = "hems_data_final.csv"
    processor = HEMSDataProcessor(data_path)
    
    print("\n1. Testing data loading...")
    raw_data = processor.load_data()
    print(f"Data loaded: {len(raw_data)} rows")
    print(f"Columns: {raw_data.columns.tolist()}")
    
    print("\n2. Testing data preprocessing...")
    processed_data = processor.preprocess_data()
    print(f"Processed data shape: {processed_data.shape}")
    print(f"New features added: {set(processed_data.columns) - set(raw_data.columns)}")
    
    print("\n3. Testing data normalization...")
    normalized_data = processor.normalize_data()
    
    normalized_features = [
        'Temperature', 'Energy_Price', 'HVAC', 'Water_Heater', 
        'Total_Consumption', 'PV_Generation', 'Grid_Carbon_Intensity',
        'Battery_SOC', 'Temperature_Deviation'
    ]
    
    print("\nNormalized features statistics:")
    for feature in normalized_features:
        mean = normalized_data[feature].mean()
        std = normalized_data[feature].std()
        print(f"{feature:<25} Mean: {mean:.4f}, Std: {std:.4f}")
    
    print("\n4. Testing data splitting...")
    train_data, test_data = processor.split_data(test_size=0.2)
    print(f"Train data: {len(train_data)} rows")
    print(f"Test data: {len(test_data)} rows")
    
    print("\n5. Testing SNN input preparation...")
    train_loader, test_loader, input_size = processor.prepare_snn_input(batch_size=64)
    print(f"Input size: {input_size}")
    print(f"Feature names: {processor.feature_names}")
    
    print("\n6. Testing pattern analysis...")
    patterns = processor.analyze_data_patterns()
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    patterns['hourly_consumption'].plot()
    plt.title('Hourly Energy Consumption')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Consumption (kWh)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    patterns['hourly_pv'].plot()
    plt.title('Hourly PV Generation')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Generation (kWh)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    patterns['hourly_price'].plot()
    plt.title('Hourly Energy Price')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Price')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    patterns['battery_usage'].plot()
    plt.title('Battery Usage Patterns')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Power (kW)')
    plt.grid(True)
    plt.legend(['Charge', 'Discharge'])
    
    plt.tight_layout()
    plt.savefig('data_patterns.png')
    
    print(f"\nTemperature-HVAC correlation: {patterns['temp_hvac_corr']:.4f}")
    
    print("\n7. Testing feature statistics...")
    stats = processor.get_feature_stats()
    
    key_features = ['Temperature', 'Energy_Price', 'HVAC', 'PV_Generation', 'Battery_SOC']
    print("\nKey feature statistics:")
    for feature in key_features:
        if feature in stats:
            print(f"{feature:<15} Min: {stats[feature]['min']:<8.4f} Max: {stats[feature]['max']:<8.4f} Mean: {stats[feature]['mean']:<8.4f}")
    
    print("\nAll tests completed successfully!")
    return processor

if __name__ == "__main__":
    processor = test_data_processor()
    plt.show() 
    
    print("\nGenerating additional visualizations...")
    
    key_features = ['Temperature', 'Energy_Price', 'HVAC', 'PV_Generation', 
                    'Total_Consumption', 'Grid_Carbon_Intensity']
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(key_features):
        plt.subplot(2, 3, i+1)
        sns.histplot(processor.processed_data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    
    plt.figure(figsize=(15, 8))
    
    sample = processor.processed_data.iloc[:168]  # First week (24*7 hours)
    
    plt.subplot(2, 1, 1)
    plt.plot(sample['Timestamp'], sample['Total_Consumption'], label='Consumption')
    plt.plot(sample['Timestamp'], sample['PV_Generation'], label='PV Generation')
    plt.title('Energy Consumption and Generation (1 Week)')
    plt.xlabel('Time')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(sample['Timestamp'], sample['Energy_Price'], label='Energy Price')
    plt.plot(sample['Timestamp'], sample['Temperature'], label='Temperature')
    plt.title('Energy Price and Temperature (1 Week)')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('time_series.png')
    
    print("Visualizations saved to 'data_patterns.png', 'feature_distributions.png', and 'time_series.png'")
    plt.show() 
