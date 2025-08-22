import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import mne
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import tempfile
import os
from scipy import signal
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
import pywt
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="EEG Schizophrenia Classifier",
    page_icon="🧠",
    layout="wide"
)

# EXACT Model Architecture (matching training script)
class EEGNet(nn.Module):  # Using same name as training script
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

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model, scaler, and metadata"""
    try:
        # Load metadata
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load model with EXACT same architecture
        model = EEGNet(metadata['input_dim'])
        model.load_state_dict(torch.load('eeg_schizophrenia_model.pt', map_location='cpu'))
        model.eval()
        
        # Load scaler
        scaler = joblib.load('eeg_scaler.pkl')
        
        return model, scaler, metadata
    except FileNotFoundError as e:
        st.error(f"Required files not found: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def extract_wavelet_features(window, wavelet='db4', level=4):
    """Extract wavelet features - EXACT match to training"""
    features = []
    for ch in window:
        coeffs = pywt.wavedec(ch, wavelet, level=level)
        # Use mean/std of detail coefficients as features
        for c in coeffs[1:]:  # skip approximation
            if len(c) > 0:
                features.extend([np.mean(c), np.std(c)])
            else:
                features.extend([0.0, 0.0])
    return np.array(features)

def extract_fft_features(window):
    """Extract FFT features - EXACT match to training"""
    features = []
    for ch in window:
        f = np.abs(fft(ch))
        features.extend([np.mean(f), np.std(f), np.max(f), np.median(f)])
    return np.array(features)

def window_signal(sig, window_size=500, step=250):
    """Create overlapping windows - EXACT match to training"""
    windows = []
    for start in range(0, sig.shape[1] - window_size + 1, step):
        window = sig[:, start:start+window_size]
        windows.append(window)
    return np.array(windows)

def load_edf_file(file_bytes):
    """Load EDF file from uploaded bytes"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name
        
        raw = mne.io.read_raw_edf(tmp_file_path, preload=True, verbose=False)
        os.unlink(tmp_file_path)
        return raw
    except Exception as e:
        st.error(f"Error loading EDF file: {str(e)}")
        return None

def preprocess_eeg_data(raw, scaler, metadata, duration=4.0):
    """Preprocess EEG data - EXACT match to training pipeline"""
    try:
        # Get data from first 19 channels max (same as training)
        data = raw.get_data()
        data = data[:min(19, len(raw.ch_names)), :]
        
        # Apply EXACT same filtering as training
        raw_filtered = raw.copy()
        raw_filtered.filter(0.5, 40.0, verbose=False)
        raw_filtered.notch_filter(50, verbose=False)
        data = raw_filtered.get_data()[:min(19, len(raw.ch_names)), :]
        
        # Create windows (same as training)
        windows = window_signal(data, window_size=500, step=250)
        
        if len(windows) == 0:
            st.error("No windows could be created!")
            return None, None, None, None
        
        # Extract features from ALL windows (same as training)
        all_features = []
        for window in windows:
            # Pad channels to 19 if needed (same as training)
            if window.shape[0] < 19:
                padding = np.zeros((19 - window.shape[0], window.shape[1]))
                window = np.concatenate([window, padding], axis=0)
            
            # Extract features
            wavelet_feats = extract_wavelet_features(window)
            fft_feats = extract_fft_features(window)
            features = np.concatenate([wavelet_feats, fft_feats])
            all_features.append(features)
        
        # IMPORTANT: Use first window features (same as training approach)
        final_features = all_features[0]
        
        # Apply scaler
        final_features_scaled = scaler.transform([final_features])[0]
        
        # Return minimal data to avoid serialization issues
        return final_features_scaled, raw.ch_names[:min(19, len(raw.ch_names))], raw_filtered.info['sfreq'], None
    
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None, None, None, None

def make_prediction(model, features):
    """Make prediction with proper confidence calculation"""
    try:
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(features_tensor)
            # Use temperature scaling to get more realistic probabilities
            probabilities = torch.softmax(outputs / 1.5, dim=1)  # Temperature = 1.5
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].numpy()
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def create_simple_plot(data_sample, channels, title="EEG Sample"):
    """Create a simple plot to avoid serialization issues"""
    # Use only first 1000 points to avoid large data serialization
    sample_size = min(1000, data_sample.shape[1]) if data_sample is not None else 1000
    
    fig = go.Figure()
    
    if data_sample is not None:
        time = np.linspace(0, sample_size/250, sample_size)  # Assume 250Hz
        for i, channel in enumerate(channels[:min(5, len(channels))]):  # Show max 5 channels
            if i < data_sample.shape[0]:
                fig.add_trace(go.Scatter(
                    x=time,
                    y=data_sample[i, :sample_size],
                    name=channel,
                    line=dict(width=1)
                ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=400
    )
    
    return fig

def main():
    st.title("🧠 EEG Schizophrenia Classification")
    st.markdown("Upload an EDF file to classify EEG signals for schizophrenia detection.")
    
    # Load model, scaler, and metadata
    model, scaler, metadata = load_model_and_scaler()
    if model is None:
        st.error("Could not load model files. Please ensure you have:")
        st.error("- eeg_schizophrenia_model.pt")
        st.error("- eeg_scaler.pkl") 
        st.error("- model_metadata.json")
        st.stop()
    
    # Display model info in sidebar
    st.sidebar.header("Model Information")
    st.sidebar.write(f"**Input Features:** {metadata['input_dim']}")
    st.sidebar.write(f"**Training Accuracy:** {metadata['accuracy']:.2f}%")
    st.sidebar.write(f"**Classes:** {', '.join(metadata['class_names'])}")
    
    # Parameters
    st.sidebar.header("Settings")
    show_details = st.sidebar.checkbox("Show Processing Details", False)
    
    # File upload
    uploaded_file = st.file_uploader("Upload EDF File", type=['edf'])
    
    if uploaded_file is not None:
        st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        # Load EDF file
        with st.spinner("Loading EDF file..."):
            raw = load_edf_file(uploaded_file.read())
        
        if raw is not None:
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sampling Rate", f"{raw.info['sfreq']:.0f} Hz")
            with col2:
                st.metric("Channels", len(raw.ch_names))
            with col3:
                st.metric("Duration", f"{raw.times[-1]:.1f} sec")
            
            # Show available channels
            if show_details:
                st.write("**Available Channels:**", ', '.join(raw.ch_names))
            
            # Preprocess data
            with st.spinner("Preprocessing EEG data..."):
                processed_features, selected_channels, sfreq, _ = preprocess_eeg_data(raw, scaler, metadata)
            
            if processed_features is not None:
                if show_details:
                    st.success(f"✅ Preprocessed: {len(processed_features)} features extracted")
                    st.write(f"**Selected Channels:** {', '.join(selected_channels)}")
                    st.write(f"**Feature Range:** {processed_features.min():.4f} to {processed_features.max():.4f}")
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    predicted_class, confidence, probabilities = make_prediction(model, processed_features)
                
                if predicted_class is not None:
                    # Display results
                    st.subheader("🔍 Classification Results")
                    
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        class_names = metadata['class_names']
                        prediction_text = class_names[predicted_class]
                        
                        # Color coding
                        if predicted_class == 0:  # Healthy
                            color = "green"
                            emoji = "✅"
                        else:  # Schizophrenia
                            color = "red"
                            emoji = "⚠️"
                        
                        st.markdown(f"### {emoji} **{prediction_text}**")
                        st.markdown(f"**Confidence:** {confidence:.1%}")
                        
                        # Confidence interpretation
                        if confidence > 0.8:
                            conf_level = "Very High"
                        elif confidence > 0.6:
                            conf_level = "High"
                        elif confidence > 0.5:
                            conf_level = "Moderate"
                        else:
                            conf_level = "Low"
                        
                        st.write(f"**Confidence Level:** {conf_level}")
                    
                    with col2:
                        # Probability visualization
                        prob_df = pd.DataFrame({
                            'Class': class_names,
                            'Probability': probabilities
                        })
                        
                        fig_prob = px.bar(
                            prob_df, 
                            x='Class', 
                            y='Probability',
                            title="Classification Probabilities",
                            color='Probability',
                            color_continuous_scale="RdYlGn_r",
                            text='Probability'
                        )
                        fig_prob.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                        fig_prob.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Feature analysis
                    if show_details:
                        st.subheader("📊 Feature Analysis")
                        
                        # Feature statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Feature Count", len(processed_features))
                        with col2:
                            st.metric("Mean", f"{np.mean(processed_features):.3f}")
                        with col3:
                            st.metric("Std Dev", f"{np.std(processed_features):.3f}")
                        with col4:
                            st.metric("Range", f"{processed_features.max() - processed_features.min():.3f}")
                        
                        # Feature distribution (show only first 50 to avoid issues)
                        fig_feat = px.histogram(
                            x=processed_features[:50], 
                            nbins=20,
                            title="Feature Distribution (First 50 Features)",
                            labels={'x': 'Feature Value', 'y': 'Count'}
                        )
                        fig_feat.update_layout(height=300)
                        st.plotly_chart(fig_feat, use_container_width=True)
                    
                    # Simple EEG visualization (avoid large data serialization)
                    try:
                        sample_data = raw.get_data()[:5, :1000]  # First 5 channels, 1000 points
                        fig_eeg = create_simple_plot(sample_data, selected_channels[:5], "EEG Sample (First 4 seconds)")
                        st.subheader("📈 EEG Signal Sample")
                        st.plotly_chart(fig_eeg, use_container_width=True)
                    except:
                        st.info("EEG visualization not available")
    
    # Instructions
    st.markdown("---")
    st.subheader("ℹ️ How to Use")
    st.markdown("""
    1. **Upload** an EDF file containing EEG recordings
    2. **Wait** for preprocessing (filtering, windowing, feature extraction)
    3. **View** the classification result and confidence score
    4. **Enable** "Show Processing Details" for more information
    
    **Note:** The model classifies EEG signals as either Healthy or Schizophrenia based on 
    wavelet and frequency domain features extracted from the signal.
    """)
    
    # Technical info
    with st.expander("Technical Details"):
        st.markdown("""
        **Preprocessing Pipeline:**
        - Bandpass filter: 0.5-40 Hz
        - Notch filter: 50 Hz (power line noise)
        - Windowing: 2-second windows, 1-second overlap
        - Feature extraction: Wavelet (Daubechies db4) + FFT features
        - Normalization: StandardScaler fitted during training
        
        **Model Architecture:**
        - Input: Feature vector
        - Hidden layers: 128 → 32 neurons (ReLU activation)
        - Output: 2 classes (Healthy, Schizophrenia)
        - Dropout: 30% during training
        """)

if __name__ == "__main__":
    main()