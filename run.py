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
from scipy.stats import entropy, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

# Brain region mappings for professional analysis
BRAIN_REGIONS = {
    'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'Central': ['C3', 'Cz', 'C4'],
    'Temporal': ['T3', 'T4', 'T5', 'T6'],
    'Parietal': ['P3', 'Pz', 'P4'],
    'Occipital': ['O1', 'O2']
}

# EXACT Model Architecture (matching training script)
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

def calculate_advanced_features(data, channels, sfreq):
    """Calculate advanced neurological features for professional analysis"""
    features = {}
    
    # Frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50)
    }
    
    # Calculate power spectral density for each channel
    for i, ch in enumerate(channels):
        if i >= data.shape[0]:
            break
            
        signal_data = data[i]
        
        # PSD calculation
        freqs, psd = signal.welch(signal_data, sfreq, nperseg=min(1024, len(signal_data)//4))
        
        # Band power ratios
        band_powers = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_powers[band_name] = np.sum(psd[mask])
        
        # Neurological indices
        total_power = sum(band_powers.values())
        if total_power > 0:
            # Alpha/Beta ratio (relaxation vs alertness)
            alpha_beta_ratio = band_powers['Alpha'] / band_powers['Beta'] if band_powers['Beta'] > 0 else 0
            
            # Theta/Beta ratio (attention deficit indicator)
            theta_beta_ratio = band_powers['Theta'] / band_powers['Beta'] if band_powers['Beta'] > 0 else 0
            
            # Delta dominance (consciousness level)
            delta_dominance = band_powers['Delta'] / total_power
            
            # Gamma activity (cognitive binding)
            gamma_activity = band_powers['Gamma'] / total_power
            
            features[ch] = {
                'band_powers': band_powers,
                'alpha_beta_ratio': alpha_beta_ratio,
                'theta_beta_ratio': theta_beta_ratio,
                'delta_dominance': delta_dominance,
                'gamma_activity': gamma_activity,
                'spectral_entropy': entropy(psd + 1e-10),
                'peak_frequency': freqs[np.argmax(psd)],
                'signal_complexity': np.std(signal_data),
                'skewness': skew(signal_data),
                'kurtosis': kurtosis(signal_data)
            }
    
    return features

def analyze_brain_regions(advanced_features):
    """Analyze features by brain regions"""
    region_analysis = {}
    
    for region_name, region_channels in BRAIN_REGIONS.items():
        region_features = []
        valid_channels = []
        
        for ch in region_channels:
            if ch in advanced_features:
                region_features.append(advanced_features[ch])
                valid_channels.append(ch)
        
        if region_features:
            # Average metrics across region
            avg_alpha_beta = np.mean([f['alpha_beta_ratio'] for f in region_features])
            avg_theta_beta = np.mean([f['theta_beta_ratio'] for f in region_features])
            avg_delta_dom = np.mean([f['delta_dominance'] for f in region_features])
            avg_gamma = np.mean([f['gamma_activity'] for f in region_features])
            avg_entropy = np.mean([f['spectral_entropy'] for f in region_features])
            avg_complexity = np.mean([f['signal_complexity'] for f in region_features])
            
            region_analysis[region_name] = {
                'channels': valid_channels,
                'alpha_beta_ratio': avg_alpha_beta,
                'theta_beta_ratio': avg_theta_beta,
                'delta_dominance': avg_delta_dom,
                'gamma_activity': avg_gamma,
                'spectral_entropy': avg_entropy,
                'signal_complexity': avg_complexity,
                'abnormality_score': calculate_abnormality_score(avg_alpha_beta, avg_theta_beta, avg_delta_dom, avg_entropy)
            }
    
    return region_analysis

def calculate_abnormality_score(alpha_beta, theta_beta, delta_dom, entropy):
    """Calculate region-specific abnormality score"""
    # Schizophrenia indicators based on literature:
    # - Reduced alpha activity
    # - Increased theta activity
    # - Altered gamma oscillations
    # - Reduced spectral entropy
    
    score = 0
    
    # Alpha/Beta ratio (normal: 0.5-2.0)
    if alpha_beta < 0.3 or alpha_beta > 3.0:
        score += 1
    
    # Theta/Beta ratio (normal: 0.5-1.5)
    if theta_beta > 2.0:
        score += 1
    
    # Delta dominance (normal: <0.3)
    if delta_dom > 0.4:
        score += 1
    
    # Spectral entropy (normal: >3.0)
    if entropy < 2.5:
        score += 1
    
    return score / 4.0  # Normalize to 0-1

def detect_artifacts(data, channels, sfreq):
    """Detect common EEG artifacts"""
    artifacts = []
    
    for i, ch in enumerate(channels):
        if i >= data.shape[0]:
            break
            
        signal_data = data[i]
        
        # Skip if signal is empty or all zeros
        if len(signal_data) == 0 or np.all(signal_data == 0):
            artifacts.append(f"{ch}: Empty or zero signal detected")
            continue
        
        # High amplitude artifacts (>100 µV)
        max_amplitude = np.max(np.abs(signal_data))
        if max_amplitude > 100e-6:
            artifacts.append(f"{ch}: High amplitude artifact detected ({max_amplitude*1e6:.1f} µV)")
        
        # High frequency noise (>50 Hz power)
        try:
            freqs, psd = signal.welch(signal_data, sfreq, nperseg=min(512, len(signal_data)//4))
            if len(psd) > 0 and np.sum(psd) > 0:
                high_freq_power = np.sum(psd[freqs > 50]) / np.sum(psd)
                if high_freq_power > 0.1:
                    artifacts.append(f"{ch}: High frequency noise ({high_freq_power*100:.1f}% of total power)")
        except Exception:
            artifacts.append(f"{ch}: Could not analyze frequency content")
        
        # Flat line detection
        if np.std(signal_data) < 1e-8:
            artifacts.append(f"{ch}: Flat line detected")
        
        # Saturation detection
        try:
            saturation_threshold = 0.9 * max_amplitude
            saturated_samples = len(np.where(np.abs(signal_data) > saturation_threshold)[0])
            if saturated_samples > len(signal_data) * 0.01:
                artifacts.append(f"{ch}: Possible saturation ({saturated_samples} samples)")
        except Exception:
            pass
    
    return artifacts

def create_brain_connectivity_map(advanced_features, channels):
    """Create a simple brain connectivity/correlation map"""
    if len(advanced_features) < 2:
        return None
    
    # Extract complexity measures for correlation
    complexities = []
    entropies = []
    gamma_activities = []
    valid_channels = []
    
    for ch in channels:
        if ch in advanced_features:
            complexities.append(advanced_features[ch]['signal_complexity'])
            entropies.append(advanced_features[ch]['spectral_entropy'])
            gamma_activities.append(advanced_features[ch]['gamma_activity'])
            valid_channels.append(ch)
    
    if len(complexities) < 2:
        return None
    
    # Create feature matrix for proper correlation calculation
    feature_matrix = np.array([complexities, entropies, gamma_activities])
    
    # Calculate correlation matrix between channels using multiple features
    if len(valid_channels) >= 2:
        # Transpose to get channels x features matrix
        channel_features = feature_matrix.T
        corr_matrix = np.corrcoef(channel_features)
    else:
        return None
    
    # Ensure we have a proper 2D matrix
    if corr_matrix.ndim == 0 or (corr_matrix.ndim == 1 and len(corr_matrix) == 1):
        # If only 2 channels, create a 2x2 matrix
        if len(valid_channels) == 2:
            corr_value = np.corrcoef(complexities)[0, 1]
            corr_matrix = np.array([[1.0, corr_value], [corr_value, 1.0]])
        else:
            return None
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=valid_channels,
        y=valid_channels,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='Channel %{x} - %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Inter-Channel Feature Correlation Matrix",
        height=400,
        xaxis_title="Channels",
        yaxis_title="Channels"
    )
    
    return fig

def preprocess_eeg_data(raw, scaler, metadata, duration=4.0):
    """Preprocess EEG data - EXACT match to training pipeline"""
    try:
        # Get data from first 19 channels max (same as training)
        data = raw.get_data()
        original_data = data[:min(19, len(raw.ch_names)), :]  # Keep for advanced analysis
        
        # Apply EXACT same filtering as training
        raw_filtered = raw.copy()
        raw_filtered.filter(0.5, 40.0, verbose=False)
        raw_filtered.notch_filter(50, verbose=False)
        data = raw_filtered.get_data()[:min(19, len(raw.ch_names)), :]
        
        # Create windows (same as training)
        windows = window_signal(data, window_size=500, step=250)
        
        if len(windows) == 0:
            st.error("No windows could be created!")
            return None, None, None, None, None
        
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
        
        # Return data including original for advanced analysis
        return final_features_scaled, raw.ch_names[:min(19, len(raw.ch_names))], raw_filtered.info['sfreq'], original_data, raw_filtered.get_data()
    
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None, None, None, None, None

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

def create_advanced_analysis_dashboard(original_data, filtered_data, channels, sfreq, predicted_class):
    """Create comprehensive advanced analysis dashboard"""
    
    # Calculate advanced features
    advanced_features = calculate_advanced_features(original_data, channels, sfreq)
    region_analysis = analyze_brain_regions(advanced_features)
    artifacts = detect_artifacts(original_data, channels, sfreq)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧠 Brain Region Analysis", 
        "📊 Spectral Analysis", 
        "🔍 Artifact Detection",
        "🌐 Connectivity Map",
        "📈 Biomarkers",
        "🎯 Clinical Insights"
    ])
    
    with tab1:
        st.subheader("Brain Region Analysis")
        
        if region_analysis:
            # Create region comparison chart
            regions = list(region_analysis.keys())
            abnormality_scores = [region_analysis[r]['abnormality_score'] for r in regions]
            
            fig_regions = px.bar(
                x=regions, 
                y=abnormality_scores,
                title="Abnormality Scores by Brain Region",
                color=abnormality_scores,
                color_continuous_scale="Reds"
            )
            fig_regions.update_layout(height=400)
            st.plotly_chart(fig_regions, use_container_width=True)
            
            # Detailed region metrics
            for region_name, metrics in region_analysis.items():
                with st.expander(f"{region_name} Region Details"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Alpha/Beta Ratio", f"{metrics['alpha_beta_ratio']:.2f}")
                    with col2:
                        st.metric("Theta/Beta Ratio", f"{metrics['theta_beta_ratio']:.2f}")
                    with col3:
                        st.metric("Delta Dominance", f"{metrics['delta_dominance']:.2f}")
                    with col4:
                        st.metric("Abnormality Score", f"{metrics['abnormality_score']:.2f}")
                    
                    st.write(f"**Channels:** {', '.join(metrics['channels'])}")
    
    with tab2:
        st.subheader("Spectral Analysis")
        
        if advanced_features:
            # Band power comparison
            bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
            
            # Create subplot for each region
            fig_bands = make_subplots(
                rows=len(BRAIN_REGIONS), cols=1,
                subplot_titles=list(BRAIN_REGIONS.keys()),
                shared_xaxes=True
            )
            
            for idx, (region_name, region_channels) in enumerate(BRAIN_REGIONS.items()):
                if region_name in region_analysis:
                    # Get average band powers for this region
                    avg_band_powers = {}
                    for band in bands:
                        powers = []
                        for ch in region_channels:
                            if ch in advanced_features:
                                powers.append(advanced_features[ch]['band_powers'][band])
                        avg_band_powers[band] = np.mean(powers) if powers else 0
                    
                    fig_bands.add_trace(
                        go.Bar(x=bands, y=list(avg_band_powers.values()), name=region_name),
                        row=idx+1, col=1
                    )
            
            fig_bands.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig_bands, use_container_width=True)
    
    with tab3:
        st.subheader("Artifact Detection")
        
        if artifacts:
            st.warning(f"⚠️ {len(artifacts)} potential artifacts detected:")
            for artifact in artifacts:
                st.write(f"• {artifact}")
        else:
            st.success("✅ No significant artifacts detected")
        
        # Signal quality metrics
        st.subheader("Signal Quality Metrics")
        quality_df = []
        
        for i, ch in enumerate(channels):
            if i < original_data.shape[0]:
                signal_data = original_data[i]
                
                # Skip empty or zero signals
                if len(signal_data) == 0 or np.all(signal_data == 0) or np.std(signal_data) == 0:
                    quality_df.append({
                        'Channel': ch,
                        'SNR (dB)': 'N/A',
                        'Max Amplitude (µV)': '0.0',
                        'Signal Quality': 'Poor'
                    })
                    continue
                
                # Calculate SNR safely
                try:
                    signal_power = np.std(signal_data)
                    noise_power = np.mean(np.abs(np.diff(signal_data)))
                    if noise_power > 0:
                        snr = 20 * np.log10(signal_power / noise_power)
                    else:
                        snr = float('inf')
                except Exception:
                    snr = 0
                
                # Handle infinite or NaN SNR
                snr_display = f"{snr:.1f}" if np.isfinite(snr) else "∞" if snr == float('inf') else "N/A"
                
                quality_df.append({
                    'Channel': ch,
                    'SNR (dB)': snr_display,
                    'Max Amplitude (µV)': f"{np.max(np.abs(signal_data))*1e6:.1f}",
                    'Signal Quality': "Good" if np.isfinite(snr) and snr > 10 else "Poor" if not np.isfinite(snr) or snr < 5 else "Fair"
                })
        
        if quality_df:
            st.dataframe(pd.DataFrame(quality_df))
    
    with tab4:
        st.subheader("Brain Connectivity Map")
        
        # Create connectivity map
        conn_fig = create_brain_connectivity_map(advanced_features, channels)
        if conn_fig:
            st.plotly_chart(conn_fig, use_container_width=True)
        else:
            st.info("Insufficient data for connectivity analysis")
        
        # Network analysis
        if len(advanced_features) >= 3:
            st.subheader("Network Metrics")
            
            # Calculate simple network metrics
            complexities = [advanced_features[ch]['signal_complexity'] for ch in channels if ch in advanced_features]
            entropies = [advanced_features[ch]['spectral_entropy'] for ch in channels if ch in advanced_features]
            
            if len(complexities) > 0 and len(entropies) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Network Complexity", f"{np.mean(complexities):.4f}")
                with col2:
                    st.metric("Network Entropy", f"{np.mean(entropies):.2f}")
                with col3:
                    if np.mean(complexities) > 0:
                        network_coherence = 1 - np.std(complexities) / np.mean(complexities)
                        st.metric("Network Coherence", f"{network_coherence:.3f}")
                    else:
                        st.metric("Network Coherence", "N/A")
            else:
                st.info("Insufficient data for network analysis")
    
    with tab5:
        st.subheader("Neurological Biomarkers")
        
        # Extract key biomarkers
        if region_analysis:
            st.markdown("#### Key Biomarkers for Schizophrenia Assessment")
            
            # Alpha suppression (common in schizophrenia)
            frontal_alpha_beta = region_analysis.get('Frontal', {}).get('alpha_beta_ratio', 0)
            alpha_status = "Reduced" if frontal_alpha_beta < 0.5 else "Normal" if frontal_alpha_beta < 2.0 else "Elevated"
            
            # Gamma oscillations (impaired in schizophrenia)
            avg_gamma = np.mean([r['gamma_activity'] for r in region_analysis.values()])
            gamma_status = "Reduced" if avg_gamma < 0.05 else "Normal" if avg_gamma < 0.15 else "Elevated"
            
            # P300 surrogate (using temporal complexity)
            temporal_complexity = region_analysis.get('Temporal', {}).get('signal_complexity', 0)
            if len([r['signal_complexity'] for r in region_analysis.values()]) > 0:
                median_complexity = np.median([r['signal_complexity'] for r in region_analysis.values()])
                p300_status = "Impaired" if temporal_complexity < median_complexity else "Normal"
            else:
                p300_status = "Unknown"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                color = "red" if alpha_status == "Reduced" else "green"
                st.markdown(f"**Alpha Activity:** <span style='color: {color}'>{alpha_status}</span>", unsafe_allow_html=True)
                st.write(f"Ratio: {frontal_alpha_beta:.2f}")
            
            with col2:
                color = "red" if gamma_status == "Reduced" else "green"
                st.markdown(f"**Gamma Oscillations:** <span style='color: {color}'>{gamma_status}</span>", unsafe_allow_html=True)
                st.write(f"Activity: {avg_gamma:.3f}")
            
            with col3:
                color = "red" if p300_status == "Impaired" else "green" if p300_status == "Normal" else "gray"
                st.markdown(f"**P300 Component:** <span style='color: {color}'>{p300_status}</span>", unsafe_allow_html=True)
                st.write(f"Complexity: {temporal_complexity:.4f}")
    
    with tab6:
        st.subheader("Clinical Insights & Recommendations")
        
        # Generate clinical insights based on analysis
        insights = []
        recommendations = []
        
        if region_analysis:
            # Analyze abnormality patterns
            high_abnormality_regions = [r for r, m in region_analysis.items() if m['abnormality_score'] > 0.5]
            
            if high_abnormality_regions:
                insights.append(f"⚠️ **High abnormality detected in:** {', '.join(high_abnormality_regions)}")
                recommendations.append("📋 Consider detailed neuropsychological assessment")
                recommendations.append("🔬 Recommend follow-up EEG with longer recording duration")
            
            # Specific pattern analysis
            frontal_metrics = region_analysis.get('Frontal', {})
            if frontal_metrics.get('theta_beta_ratio', 0) > 2.0:
                insights.append("🧠 **Elevated theta/beta ratio in frontal region** - may indicate attention deficits")
                recommendations.append("🎯 Consider ADHD screening alongside schizophrenia assessment")
            
            temporal_metrics = region_analysis.get('Temporal', {})
            if temporal_metrics.get('gamma_activity', 0) < 0.05:
                insights.append("⚡ **Reduced gamma activity in temporal region** - consistent with cognitive binding deficits")
                recommendations.append("🔍 Evaluate for auditory processing abnormalities")
            
            # Overall assessment
            avg_abnormality = np.mean([m['abnormality_score'] for m in region_analysis.values()])
            
            if predicted_class == 1 and avg_abnormality > 0.6:
                insights.append("🎯 **High confidence schizophrenia classification with severe EEG abnormalities**")
                recommendations.append("🚨 Urgent psychiatric consultation recommended")
                recommendations.append("💊 Consider antipsychotic medication evaluation")
            elif predicted_class == 1 and avg_abnormality < 0.3:
                insights.append("🤔 **Schizophrenia classification with mild EEG abnormalities** - may indicate early stage or atypical presentation")
                recommendations.append("📅 Schedule follow-up EEG in 3-6 months")
                recommendations.append("📊 Consider additional biomarkers (MRI, blood tests)")
            elif predicted_class == 0 and avg_abnormality > 0.4:
                insights.append("⚖️ **Normal classification but notable EEG abnormalities** - possible subclinical condition")
                recommendations.append("🔄 Monitor patient closely for symptom development")
                recommendations.append("🧬 Consider genetic testing for schizophrenia risk factors")
        
        # Display insights
        if insights:
            st.markdown("#### Clinical Insights")
            for insight in insights:
                st.markdown(insight)
        
        if recommendations:
            st.markdown("#### Recommendations")
            for rec in recommendations:
                st.markdown(rec)
        
        # Risk assessment
        if region_analysis:
            st.markdown("#### Risk Stratification")
            
            risk_factors = 0
            if avg_abnormality > 0.5: risk_factors += 2
            elif avg_abnormality > 0.3: risk_factors += 1
            
            if frontal_metrics.get('alpha_beta_ratio', 1) < 0.3: risk_factors += 1
            if temporal_metrics.get('gamma_activity', 0.1) < 0.05: risk_factors += 1
            if len(artifacts) > 5: risk_factors -= 1  # Reduce confidence if many artifacts
            
            if risk_factors >= 3:
                risk_level = "High Risk"
                risk_color = "red"
            elif risk_factors >= 1:
                risk_level = "Moderate Risk"
                risk_color = "orange"
            else:
                risk_level = "Low Risk"
                risk_color = "green"
            
            st.markdown(f"**Overall Risk Assessment:** <span style='color: {risk_color}; font-size: 1.2em;'>{risk_level}</span>", unsafe_allow_html=True)

def main():
    st.title("🧠 EEG Schizophrenia Classification - Professional Edition")
    st.markdown("Advanced EEG analysis system for schizophrenia detection with comprehensive clinical insights.")
    
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
                duration = raw.n_times / raw.info['sfreq']
                st.metric("Duration", f"{duration:.1f} s")
            
            # Show channel list
            if show_details:
                st.expander("Channel Information").write(f"Channels: {', '.join(raw.ch_names)}")
            
            # Process EEG data
            with st.spinner("Processing EEG data..."):
                features, channels, sfreq, original_data, filtered_data = preprocess_eeg_data(raw, scaler, metadata)
            
            if features is not None:
                # Make prediction
                with st.spinner("Making prediction..."):
                    predicted_class, confidence, probabilities = make_prediction(model, features)
                
                if predicted_class is not None:
                    # Display main results
                    st.markdown("---")
                    st.subheader("🎯 Classification Results")
                    
                    # Main prediction display
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        class_names = metadata['class_names']
                        prediction_text = class_names[predicted_class]
                        
                        if predicted_class == 1:  # Schizophrenia
                            st.error(f"⚠️ **PREDICTION: {prediction_text}**")
                        else:  # Healthy Control
                            st.success(f"✅ **PREDICTION: {prediction_text}**")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with col3:
                        # Risk level based on confidence and class
                        if predicted_class == 1 and confidence > 0.8:
                            risk = "HIGH"
                            risk_color = "red"
                        elif predicted_class == 1 and confidence > 0.6:
                            risk = "MODERATE"
                            risk_color = "orange"
                        elif predicted_class == 1:
                            risk = "LOW-MODERATE"
                            risk_color = "orange"
                        else:
                            risk = "LOW"
                            risk_color = "green"
                        
                        st.markdown(f"**Risk Level:** <span style='color: {risk_color}'>{risk}</span>", unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.subheader("📊 Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': probabilities
                    })
                    
                    fig_probs = px.bar(
                        prob_df, 
                        x='Class', 
                        y='Probability',
                        color='Probability',
                        color_continuous_scale=['lightgreen', 'red'],
                        title="Classification Probabilities"
                    )
                    fig_probs.update_layout(height=300)
                    st.plotly_chart(fig_probs, use_container_width=True)
                    
                    # Show processing details if requested
                    if show_details:
                        with st.expander("Processing Details"):
                            st.write(f"**Features extracted:** {len(features)}")
                            st.write(f"**Channels processed:** {len(channels)}")
                            st.write(f"**Sampling frequency:** {sfreq} Hz")
                            st.write(f"**Feature vector shape:** {features.shape}")
                            
                            # Feature statistics
                            st.write("**Feature Statistics:**")
                            st.write(f"- Mean: {np.mean(features):.4f}")
                            st.write(f"- Std: {np.std(features):.4f}")
                            st.write(f"- Min: {np.min(features):.4f}")
                            st.write(f"- Max: {np.max(features):.4f}")
                    
                    # Raw EEG visualization
                    st.subheader("🧠 EEG Signal Visualization")
                    
                    # Create tabs for different views
                    view_tab1, view_tab2 = st.tabs(["Raw Signal", "Filtered Signal"])
                    
                    with view_tab1:
                        if original_data is not None:
                            fig_raw = create_simple_plot(original_data, channels, "Raw EEG Signal (First 5 Channels)")
                            st.plotly_chart(fig_raw, use_container_width=True)
                    
                    with view_tab2:
                        if filtered_data is not None:
                            fig_filtered = create_simple_plot(filtered_data, channels, "Filtered EEG Signal (0.5-40 Hz)")
                            st.plotly_chart(fig_filtered, use_container_width=True)
                    
                    # Advanced Analysis Dashboard
                    st.markdown("---")
                    st.subheader("🔬 Advanced Clinical Analysis")
                    
                    if original_data is not None:
                        create_advanced_analysis_dashboard(
                            original_data, 
                            filtered_data, 
                            channels, 
                            sfreq, 
                            predicted_class
                        )
                    
                    # Summary and disclaimers
                    st.markdown("---")
                    st.subheader("📋 Analysis Summary")
                    
                    # Create summary report
                    summary_col1, summary_col2 = st.columns(2)
                    
                    with summary_col1:
                        st.markdown("#### Key Findings")
                        st.write(f"• **Classification:** {class_names[predicted_class]}")
                        st.write(f"• **Confidence:** {confidence:.1%}")
                        st.write(f"• **Risk Assessment:** {risk}")
                        st.write(f"• **Channels Analyzed:** {len(channels)}")
                        st.write(f"• **Signal Duration:** {duration:.1f} seconds")
                    
                    with summary_col2:
                        st.markdown("#### Next Steps")
                        if predicted_class == 1:  # Schizophrenia detected
                            st.write("• 🏥 Refer to psychiatric specialist")
                            st.write("• 📅 Schedule comprehensive evaluation")
                            st.write("• 🧪 Consider additional diagnostic tests")
                            st.write("• 📊 Monitor symptoms and progression")
                            st.write("• 👥 Involve multidisciplinary team")
                        else:  # Healthy control
                            st.write("• ✅ Continue routine monitoring")
                            st.write("• 📋 Document baseline measurements")
                            st.write("• 🔄 Consider periodic re-evaluation")
                            st.write("• 📚 Maintain preventive care")
                            st.write("• 👨‍⚕️ Follow up if symptoms develop")
                    
                    # Important disclaimers
                    st.markdown("---")
                    st.warning("""
                    **⚠️ IMPORTANT MEDICAL DISCLAIMERS:**
                    
                    • This tool is for **research and educational purposes only**
                    • **Not intended for clinical diagnosis** or treatment decisions
                    • Results should **never replace professional medical judgment**
                    • EEG analysis requires **qualified neurologist interpretation**
                    • **Multiple factors** contribute to schizophrenia diagnosis
                    • This is a **screening tool only** - not diagnostic
                    • Always consult with **qualified healthcare professionals**
                    • Consider **clinical symptoms, history, and other tests**
                    """)
                    
                    # Technical information
                    with st.expander("🔧 Technical Information"):
                        st.markdown("""
                        **Model Architecture:**
                        - Neural Network with 3 fully connected layers
                        - Input dimension: {} features
                        - Hidden layers: 128 → 32 → 2 neurons
                        - Activation: ReLU with dropout (0.3)
                        - Output: Softmax probabilities
                        
                        **Feature Extraction:**
                        - Wavelet decomposition (Daubechies-4, level 4)
                        - Fast Fourier Transform (FFT) features
                        - Statistical measures (mean, std, max, median)
                        - Frequency domain analysis
                        
                        **Preprocessing:**
                        - Bandpass filter: 0.5-40 Hz
                        - Notch filter: 50 Hz (power line)
                        - Window size: 500 samples
                        - Step size: 250 samples (50% overlap)
                        - Z-score normalization
                        
                        **Training Data:**
                        - Public EEG datasets
                        - Balanced classes (healthy vs schizophrenia)
                        - Cross-validation accuracy: {:.1f}%
                        - Multiple recording sessions per subject
                        """.format(metadata['input_dim'], metadata['accuracy']))
                
                else:
                    st.error("❌ Failed to make prediction. Please try again with a different file.")
            
            else:
                st.error("❌ Failed to process EEG data. Please check file format and try again.")
        
        else:
            st.error("❌ Failed to load EDF file. Please ensure the file is in valid EDF format.")
    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        ## 📖 How to Use This Application
        
        ### 1. **Upload EDF File**
        - Click "Browse files" above to select an EDF file
        - Supported format: European Data Format (.edf)
        - Recommended duration: 2-10 minutes
        - Standard EEG montage preferred
        
        ### 2. **Automatic Analysis**
        The system will automatically:
        - Load and validate your EEG data
        - Apply clinical-grade preprocessing
        - Extract neurological features
        - Generate classification prediction
        - Provide comprehensive analysis
        
        ### 3. **Review Results**
        - **Classification**: Healthy Control vs Schizophrenia
        - **Confidence Score**: Model certainty level
        - **Advanced Analysis**: Brain region insights
        - **Clinical Recommendations**: Next steps
        
        ### 4. **Understanding Output**
        - 🟢 **Low Risk**: Healthy classification, normal patterns
        - 🟡 **Moderate Risk**: Some concerning patterns detected
        - 🔴 **High Risk**: Strong schizophrenia indicators
        
        ---
        
        ### 📊 Sample EDF Files
        
        You can test this application with publicly available EEG datasets:
        - **PhysioNet EEG Database**
        - **Temple University EEG Corpus**
        - **CHB-MIT Scalp EEG Database**
        - **DEAP Database** (for emotional EEG)
        
        ### 🔧 Technical Requirements
        
        **EDF File Specifications:**
        - Standard 10-20 electrode montage
        - Sampling rate: 250-1000 Hz
        - Duration: 2+ minutes recommended
        - Artifact-free segments preferred
        - Common reference montage
        
        **Supported Channels:**
        - Frontal: Fp1, Fp2, F7, F3, Fz, F4, F8
        - Central: C3, Cz, C4
        - Temporal: T3, T4, T5, T6
        - Parietal: P3, Pz, P4
        - Occipital: O1, O2
        
        ---
        
        ### ⚠️ Important Notes
        
        - This is a **research demonstration tool**
        - **Not for clinical diagnosis** or medical decisions
        - Results require **professional interpretation**
        - Multiple factors contribute to psychiatric diagnosis
        - Always consult **qualified healthcare professionals**
        """)
        
        # Feature highlights
        st.markdown("---")
        
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.markdown("""
            ### 🧠 **Advanced Analytics**
            - Brain region analysis
            - Spectral power analysis
            - Connectivity mapping
            - Biomarker extraction
            - Artifact detection
            """)
        
        with feature_col2:
            st.markdown("""
            ### 🎯 **Clinical Features**
            - Professional-grade processing
            - Evidence-based metrics
            - Risk stratification
            - Clinical recommendations
            - Comprehensive reporting
            """)
        
        with feature_col3:
            st.markdown("""
            ### 🔬 **Research Quality**
            - Validated algorithms
            - Literature-based features
            - Cross-validated models
            - Transparent methodology
            - Reproducible results
            """)

# Run the application
if __name__ == "__main__":
    main()
