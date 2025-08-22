# EEG Schizophrenia Classification Training Script
# This recreates the exact training process from your document

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import pywt
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import joblib

# Data folder
DATA_FOLDER = './repod'  # adjust if different

# EEG channels used (from dataset description)
EEG_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
                "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]

def load_edf(file_path):
    """Load EDF file using MNE"""
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        return raw
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_file_label(file_name):
    """Get label from filename: s = schizophrenia (1), h = healthy (0)"""
    return 1 if file_name.startswith('s') else 0

def window_signal(sig, window_size=500, step=250):  # 2s windows, 1s overlap
    """Create overlapping windows from signal"""
    windows = []
    for start in range(0, sig.shape[1] - window_size + 1, step):
        window = sig[:, start:start+window_size]
        windows.append(window)
    return np.array(windows)

def extract_wavelet_features(window, wavelet='db4', level=4):
    """Extract wavelet features from EEG window"""
    features = []
    for ch in window:
        coeffs = pywt.wavedec(ch, wavelet, level=level)
        # Use mean/variance of detail coefficients as features
        for c in coeffs[1:]:  # skip approximation
            if len(c) > 0:
                features.extend([np.mean(c), np.std(c)])
            else:
                features.extend([0.0, 0.0])
    return np.array(features)

def extract_fft_features(window):
    """Extract FFT features from EEG window"""
    features = []
    for ch in window:
        f = np.abs(fft(ch))
        features.extend([np.mean(f), np.std(f), np.max(f), np.median(f)])
    return np.array(features)

# Dataset class
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Model class (exact match to your training)
class EEGNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.3, training=self.training)
        x = self.fc3(x)
        return x

def main():
    print("Starting EEG Schizophrenia Classification Training...")
    
    # List all files and show their labels
    edf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.edf')]
    labels = [get_file_label(f) for f in edf_files]
    
    print(f"Total files: {len(edf_files)} (Schizophrenia: {sum(labels)}, Healthy: {len(labels)-sum(labels)})")
    
    # Load and process data
    X = []
    y = []
    
    print("Processing EDF files...")
    for i, f in enumerate(edf_files):
        print(f"Processing {f} ({i+1}/{len(edf_files)})")
        label = get_file_label(f)
        
        raw = load_edf(os.path.join(DATA_FOLDER, f))
        if raw is None:
            continue
            
        # Get data from first 19 channels (or available channels)
        data = raw.get_data()
        available_channels = raw.ch_names
        
        # Limit to 19 channels max
        data = data[:min(19, len(available_channels)), :]
        
        # Apply basic filtering
        raw_filtered = raw.copy().filter(0.5, 40.0, verbose=False)
        raw_filtered.notch_filter(50, verbose=False)
        data = raw_filtered.get_data()[:min(19, len(available_channels)), :]
        
        # Window the signal
        windows = window_signal(data, window_size=500, step=250)
        
        print(f"  Created {len(windows)} windows from {f}")
        
        # Extract features from all windows
        for j, window in enumerate(windows):
            # Pad channels to 19 if needed
            if window.shape[0] < 19:
                padding = np.zeros((19 - window.shape[0], window.shape[1]))
                window = np.concatenate([window, padding], axis=0)
            
            # Extract features
            wavelet_feats = extract_wavelet_features(window)
            fft_feats = extract_fft_features(window)
            features = np.concatenate([wavelet_feats, fft_feats])
            
            X.append(features)
            y.append(label)
            
            if j == 0:  # Debug first window
                print(f"  Window shape: {window.shape}")
                print(f"  Wavelet features: {len(wavelet_feats)}")
                print(f"  FFT features: {len(fft_feats)}")
                print(f"  Total features: {len(features)}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nFinal dataset:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature range: {X.min():.6f} to {X.max():.6f}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Scaled feature range: {X_scaled.min():.6f} to {X_scaled.max():.6f}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train labels: {np.bincount(y_train)}, Test labels: {np.bincount(y_test)}")
    
    # Create data loaders
    train_ds = EEGDataset(X_train, y_train)
    test_ds = EEGDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EEGNet(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nModel input dimension: {X_train.shape[1]}")
    
    # Training loop
    print("\nStarting training...")
    n_epochs = 50
    train_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_dl)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Validation
    model.eval()
    correct = total = 0
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = torch.argmax(out, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.shape[0]
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())
    
    accuracy = correct/total*100
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    # Save model and scaler
    model_path = 'eeg_schizophrenia_model.pt'
    scaler_path = 'eeg_scaler.pkl'
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    # Save metadata
    metadata = {
        'input_dim': X_train.shape[1],
        'n_samples': len(X),
        'n_features': X_train.shape[1],
        'accuracy': accuracy,
        'class_names': ['Healthy', 'Schizophrenia']
    }
    
    import json
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Training completed!")
    
    # Plot some results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_true, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()