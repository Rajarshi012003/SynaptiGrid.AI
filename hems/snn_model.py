"""
SNN 
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
import matplotlib.pyplot as plt
import numpy as np

class SNN_Model(nn.Module):
    """Spiking Neural Network for HEMS control"""
    
    def __init__(self, input_size, hidden_size1=400, hidden_size2=300, output_size=3, beta=0.95):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.beta = beta  # Decay rate
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Membrane and synaptic current initialization
        self.mem1 = None
        self.mem2 = None
        self.mem3 = None
        
    def forward(self, x, num_steps=100):
        """Forward pass through time (for num_steps)"""
        
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()
        
        # Spike recording for all layers
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        mem3_rec = []  # Record final membrane potential
        
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, self.mem1 = self.lif1(cur1, self.mem1)
            
            cur2 = self.fc2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)
            
            cur3 = self.fc3(spk2)
            spk3, self.mem3 = self.lif3(cur3, self.mem3)
            
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            mem3_rec.append(self.mem3)
        
        spk1_rec = torch.stack(spk1_rec, dim=0)
        spk2_rec = torch.stack(spk2_rec, dim=0)
        spk3_rec = torch.stack(spk3_rec, dim=0)
        mem3_rec = torch.stack(mem3_rec, dim=0)
        
        outputs = torch.mean(spk3_rec, dim=0)
        
        outputs = torch.sigmoid(outputs)
        
        return outputs
    
    def encode_input(self, x, encoding_method='rate', num_steps=100):
        """Encode inputs into spike trains using various methods"""
        batch_size = x.shape[0]
        spike_trains = torch.zeros(num_steps, batch_size, self.input_size)
        
        if encoding_method == 'rate':
            x_norm = (x - x.min(dim=1, keepdim=True)[0]) / (
                x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0] + 1e-8)
            
            for t in range(num_steps):
                spike_trains[t] = torch.bernoulli(x_norm)
                
        elif encoding_method == 'temporal':
            # Temporal encoding - early spikes for high values
            # Normalize to [0, 1] and invert (so high values -> early spikes)
            x_norm = 1.0 - (x - x.min(dim=1, keepdim=True)[0]) / (
                x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0] + 1e-8)
            
            # Calculate spike times based on values (higher value = earlier spike)
            spike_times = (x_norm * num_steps).long()
            
            for b in range(batch_size):
                for i in range(self.input_size):
                    t = spike_times[b, i]
                    if t < num_steps:
                        spike_trains[t, b, i] = 1
                        
        elif encoding_method == 'direct':
            # Direct current injection - no spikes, just use values directly
            # This is handled in the forward pass, so return None
            return None
        
        return spike_trains
    
    def decode_output(self, spike_trains):
        """Decode spike trains to continuous values"""
        
        if spike_trains is not None:
            return torch.mean(spike_trains, dim=0)
        else:
            return None


def train_snn_model(model, train_loader, num_epochs=10, lr=1e-3, device='cpu', num_steps=100):
    """Train the SNN model"""
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs, num_steps=num_steps)
            
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}')
    
    return train_losses


def test_snn_model(model, test_loader, device='cpu', num_steps=100):
    """Test the SNN model"""
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs, num_steps=num_steps)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.6f}')
    
    return avg_loss


def visualize_snn_activity(model, sample_input, num_steps=100, device='cpu'):
    """Visualize SNN activity for a sample input"""
    model.to(device)
    model.eval()
    
    if not isinstance(sample_input, torch.Tensor):
        sample_input = torch.tensor(sample_input, dtype=torch.float32)
    
    if len(sample_input.shape) == 1:
        sample_input = sample_input.unsqueeze(0)
    
    sample_input = sample_input.to(device)
    
    # Initialize membrane potentials
    mem1 = model.lif1.init_leaky()
    mem2 = model.lif2.init_leaky()
    mem3 = model.lif3.init_leaky()
    
    spk1_rec = []
    spk2_rec = []
    spk3_rec = []
    mem1_rec = []
    mem2_rec = []
    mem3_rec = []
    
    for _ in range(num_steps):
        cur1 = model.fc1(sample_input)
        spk1, mem1 = model.lif1(cur1, mem1)
        
        cur2 = model.fc2(spk1)
        spk2, mem2 = model.lif2(cur2, mem2)
        
        cur3 = model.fc3(spk2)
        spk3, mem3 = model.lif3(cur3, mem3)
        
        spk1_rec.append(spk1)
        spk2_rec.append(spk2)
        spk3_rec.append(spk3)
        mem1_rec.append(mem1)
        mem2_rec.append(mem2)
        mem3_rec.append(mem3)
    
    spk3_tensor = torch.stack(spk3_rec, dim=0)
    output = torch.mean(spk3_tensor, dim=0).detach().cpu().numpy()
    
    spk1_rec = torch.stack(spk1_rec, dim=0).detach().cpu().numpy()
    spk2_rec = torch.stack(spk2_rec, dim=0).detach().cpu().numpy()
    spk3_rec = spk3_tensor.detach().cpu().numpy()
    mem1_rec = torch.stack(mem1_rec, dim=0).detach().cpu().numpy()
    mem2_rec = torch.stack(mem2_rec, dim=0).detach().cpu().numpy()
    mem3_rec = torch.stack(mem3_rec, dim=0).detach().cpu().numpy()
    
    fig, ax = plt.subplots(3, 2, figsize=(15, 10))
    
    for i in range(min(3, spk1_rec.shape[2])):
        ax[0, 0].plot(spk1_rec[:, 0, i], label=f'Neuron {i+1}')
    ax[0, 0].set_title('Layer 1 Spikes')
    ax[0, 0].set_xlabel('Time step')
    ax[0, 0].set_ylabel('Spike')
    ax[0, 0].legend()
    
    for i in range(min(3, spk2_rec.shape[2])):
        ax[1, 0].plot(spk2_rec[:, 0, i], label=f'Neuron {i+1}')
    ax[1, 0].set_title('Layer 2 Spikes')
    ax[1, 0].set_xlabel('Time step')
    ax[1, 0].set_ylabel('Spike')
    ax[1, 0].legend()
    
    for i in range(spk3_rec.shape[2]):
        ax[2, 0].plot(spk3_rec[:, 0, i], label=f'Output {i+1}')
    ax[2, 0].set_title('Output Layer Spikes')
    ax[2, 0].set_xlabel('Time step')
    ax[2, 0].set_ylabel('Spike')
    ax[2, 0].legend()
    
    for i in range(min(3, mem1_rec.shape[2])):
        ax[0, 1].plot(mem1_rec[:, 0, i], label=f'Neuron {i+1}')
    ax[0, 1].set_title('Layer 1 Membrane Potentials')
    ax[0, 1].set_xlabel('Time step')
    ax[0, 1].set_ylabel('Membrane Potential')
    ax[0, 1].legend()
    
    for i in range(min(3, mem2_rec.shape[2])):
        ax[1, 1].plot(mem2_rec[:, 0, i], label=f'Neuron {i+1}')
    ax[1, 1].set_title('Layer 2 Membrane Potentials')
    ax[1, 1].set_xlabel('Time step')
    ax[1, 1].set_ylabel('Membrane Potential')
    ax[1, 1].legend()
    
    for i in range(mem3_rec.shape[2]):
        ax[2, 1].plot(mem3_rec[:, 0, i], label=f'Output {i+1}')
    ax[2, 1].set_title('Output Layer Membrane Potentials')
    ax[2, 1].set_xlabel('Time step')
    ax[2, 1].set_ylabel('Membrane Potential')
    ax[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig('snn_activity.png')
    plt.show()
    
    print(f'Decoded outputs: {output[0]}')
    return output[0] 
