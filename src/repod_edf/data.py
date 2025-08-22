# src/repod_edf/data.py
import os, glob
import numpy as np
import mne
from mne.preprocessing import ICA, create_eog_epochs
from scipy.signal import welch
from pywt import wavedec
from dataclasses import dataclass
from typing import List, Dict, Tuple
from ..common.utils import label_from_filename


@dataclass
class EEGConfig:
    fs: int
    bandpass: Tuple[float, float]
    notch: float
    epoch_len_sec: float
    epoch_overlap: float
    use_ica: bool


class RepODLoader:
    def __init__(self, data_dir: str, cfg: EEGConfig, channels: List[str]):
        self.data_dir = data_dir
        self.cfg = cfg
        self.channels = channels

    def list_files(self):
        return sorted(glob.glob(os.path.join(self.data_dir, "*.edf")))

    def _clean_raw(self, raw: mne.io.BaseRaw):
        raw.load_data()
        raw.pick_channels(self.channels)
        raw.filter(self.cfg.bandpass[0], self.cfg.bandpass[1], fir_design='firwin')
        raw.notch_filter(self.cfg.notch)
        if self.cfg.use_ica:
            ica = ICA(n_components=min(15, len(self.channels)), random_state=97, max_iter="auto")
            ica.fit(raw)
            # try to remove EOG-like components if EOG channel absent
            try:
                eog_epochs = create_eog_epochs(raw)
                eog_inds, _ = ica.find_bads_eog(eog_epochs)
                ica.exclude = eog_inds
            except Exception:
                pass
            ica.apply(raw)
        return raw

    def epoch(self, raw: mne.io.BaseRaw):
        step = int(self.cfg.epoch_len_sec * (1 - self.cfg.epoch_overlap) * self.cfg.fs)
        size = int(self.cfg.epoch_len_sec * self.cfg.fs)
        data = raw.get_data(picks=self.channels)
        n = data.shape[1]
        epochs = []
        for start in range(0, n - size + 1, step):
            seg = data[:, start:start + size]
            epochs.append(seg)
        return np.stack(epochs) if epochs else np.empty((0, len(self.channels), size))

    def load_subject(self, fpath: str):
        raw = mne.io.read_raw_edf(fpath, preload=False, verbose=False)
        raw.info["sfreq"] = self.cfg.fs  # safety
        raw = self._clean_raw(raw)
        X = self.epoch(raw)
        y = label_from_filename(fpath)
        return X, np.full((len(X),), y, dtype=int)
