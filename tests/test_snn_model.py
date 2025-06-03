"""
Test for the SNN model
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from data_processor import HEMSDataProcessor
from snn_model import SNN_Model, train_snn_model, test_snn_model as test_snn, visualize_snn_activity

def test_snn_model():
    """Test the SNN model functionality"""
    print("Testing SNN_Model...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("\n1. Loading and preprocessing data...")
    data_path = "hems_data_final.csv"
    processor = HEMSDataProcessor(data_path)
    processor.load_data()
    processor.preprocess_data()
    processor.normalize_data()
    train_data, test_data = processor.split_data(test_size=0.2)
    
    print("\n2. Preparing data for SNN...")
    train_loader, test_loader, input_size = processor.prepare_snn_input(batch_size=64)
    print(f"Input size: {input_size}")
    print(f"Feature names: {processor.feature_names}")
    
    sample_batch = next(iter(train_loader))
    inputs, targets = sample_batch
    print(f"Sample batch shapes - inputs: {inputs.shape}, targets: {targets.shape}")
    
    print("\n3. Creating SNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    snn_model = SNN_Model(input_size, hidden_size1=64, hidden_size2=32, output_size=3)
    print(f"Model structure: {snn_model}")
    
    print("\n4. Testing forward pass...")
    sample_input = inputs[0].unsqueeze(0)  #  first input and add batch dimension
    output = snn_model(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    print("\n5. Testing spike encoding...")
    spike_trains = snn_model.encode_input(sample_input, encoding_method='rate', num_steps=10)
    if spike_trains is not None:
        print(f"Spike trains shape: {spike_trains.shape}")
        print(f"Average spike rate: {torch.mean(spike_trains):.4f}")
    
    print("\n6. Testing training for a few epochs...")
    num_epochs = 2  # Use small number for testing
    train_losses = train_snn_model(snn_model, train_loader, num_epochs=num_epochs, 
                                  lr=1e-3, device=device, num_steps=10)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses)
    plt.title(f'SNN Training Loss ({num_epochs} epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('snn_training_loss.png')
    
    print("\n7. Testing the trained model...")
    test_loss = test_snn(snn_model, test_loader, device=device, num_steps=10)
    print(f"Test loss: {test_loss:.6f}")
    
    print("\n8. Visualizing SNN activity...")
    sample_input = inputs[0].to(device)  # Get first input
    output = visualize_snn_activity(snn_model, sample_input, num_steps=10, device=device)
    
    print("\n9. Testing model with different inputs...")
    
    test_cases = [
        {"temp": 15.0, "price": 0.1, "pv": 5.0, "soc": 0.2, "carbon": 200, "hour": 12, "weather": 0},  # Sunny day, low battery
        {"temp": 25.0, "price": 0.2, "pv": 1.0, "soc": 0.8, "carbon": 400, "hour": 18, "weather": 1},  # Evening, high battery
        {"temp": 10.0, "price": 0.3, "pv": 0.0, "soc": 0.5, "carbon": 300, "hour": 22, "weather": 2},  # Night, mid battery
    ]
    
    for i, case in enumerate(test_cases):
        test_input = torch.tensor([
            case["temp"], case["price"], case["pv"], case["soc"], 
            case["carbon"], case["hour"], case["weather"], 0.0  # Temp deviation set to 0
        ], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = snn_model(test_input, num_steps=50)
        
        print(f"\nTest case {i+1}:")
        print(f"  Conditions: Temp={case['temp']}°C, Price=${case['price']}/kWh, PV={case['pv']}kW, Battery={case['soc']*100}%")
        print(f"  Time: Hour {case['hour']}, Carbon: {case['carbon']} g/kWh")
        print(f"  Outputs: HVAC={output[0,0]:.4f}, Batt_charge={output[0,1]:.4f}, Batt_discharge={output[0,2]:.4f}")
    
    print("\n10. Saving the model...")
    torch.save(snn_model.state_dict(), "snn_model_test.pth")
    
    print("\n11. Loading the model...")
    loaded_model = SNN_Model(input_size, hidden_size1=64, hidden_size2=32, output_size=3)
    loaded_model.load_state_dict(torch.load("snn_model_test.pth"))
    loaded_model.to(device)
    
    with torch.no_grad():
        original_output = snn_model(sample_input)
        loaded_output = loaded_model(sample_input)
    
    print(f"Original model output: {original_output}")
    print(f"Loaded model output: {loaded_output}")
    print(f"Outputs match: {torch.allclose(original_output, loaded_output)}")
    
    print("\nAll tests completed successfully!")
    return snn_model

if __name__ == "__main__":
    snn_model = test_snn_model()
    plt.show()  
