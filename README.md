# README.md

## Overview
Two complementary models for schizophrenia detection from EEG:

1) **RepOD EDF pipeline** (raw EEG): Clean (bandpass, notch, optional ICA) → Epoch → **Wavelet + Fourier** features → **Transformer or 1D-CNN** subject-level classifier.

2) **Kaggle Spectrogram pipeline** (images): Transfer learning (ResNet/EfficientNet/ViT) on preprocessed EEG spectrogram images.

Both save: `best.pt`, `model.onnx`, `meta.json`, plots (loss, ROC, confusion matrix, t‑SNE, Grad‑CAM).

## Quickstart
```bash
# 1) install deps (ideally in a fresh venv)
pip install -r requirements.txt

# 2) set paths in configs/*.yaml

# 3) train RepOD (Transformer by default)
bash scripts/train_repod_edf.sh

# 4) train Kaggle spectrograms
bash scripts/train_kaggle_img.sh
```

## Notes
- RepOD file naming uses `hXX.edf` for **healthy** and `sXX.edf` for **schizophrenia**; labels inferred from filename.
- Subject‑wise split avoids leakage.
- `outputs/*/model.onnx` plugs into FastAPI/Streamlit later.
- To switch RepOD model to CNN: set `MODEL.KIND: cnn1d` in `repod_edf_config.yaml`.
- For reproducibility, set a fixed `SEED`.

## FastAPI/Streamlit (later)
- Load `model.onnx` with `onnxruntime`, apply the same scaling (RepOD: use `scaler_mean.npy` & `scaler_scale.npy` to standardize features), then call inference.
- For spectrogram demo, point to any PNG and run the ONNX session.