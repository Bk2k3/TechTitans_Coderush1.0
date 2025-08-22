# src/repod_edf/features.py
import numpy as np
from scipy.signal import welch
from pywt import wavedec


def bandpower_welch(x, fs, bands):
    # x: (channels, time)
    f, Pxx = welch(x, fs=fs, nperseg=min(512, x.shape[1]), noverlap=None, axis=-1)
    bp = []
    for name, (lo, hi) in bands.items():
        idx = (f >= lo) & (f < hi)
        val = Pxx[..., idx].mean(axis=-1)
        bp.append(val)
    return np.stack(bp, axis=-1)  # (channels, n_bands)


def dwt_stats(x, wavelet="db4", level=4, stats=("energy","mean","std")):
    # x: (channels, time)
    all_stats = []
    for ch in x:
        coeffs = wavedec(ch, wavelet, level=level)
        feats = []
        for c in coeffs:
            if "energy" in stats:
                feats.append(np.sum(c**2) / len(c))
            if "mean" in stats:
                feats.append(np.mean(c))
            if "std" in stats:
                feats.append(np.std(c))
        all_stats.append(np.array(feats))
    return np.stack(all_stats, axis=0)  # (channels, n_wavelet_feats)


def extract_epoch_features(epoch, fs, bands, wavelet="db4", level=4, stats=("energy","mean","std")):
    bp = bandpower_welch(epoch, fs, bands)           # (C, B)
    wt = dwt_stats(epoch, wavelet, level, stats)     # (C, W)
    return np.concatenate([bp, wt], axis=-1)         # (C, B+W)
