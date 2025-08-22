# src/repod_edf/train.py
import os, glob, yaml, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from ..common.utils import set_seed, ensure_dir, save_json, split_by_subject
from ..common.metrics import compute_metrics
from ..common import viz
from .data import RepODLoader, EEGConfig
from .features import extract_epoch_features
from .model_cnn1d import CNN1DClassifier
from .model_transformer import TransformerClassifier


class EEGFeatureDataset(Dataset):
    def __init__(self, items):
        self.items = items  # list of (feat_tensor, label)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate_pad(batch):
    # for transformer: pad along time dimension (epochs-as-tokens)
    feats, labels = zip(*batch)
    # feats: list of (T, F) tensors; labels: ints
    T_max = max(f.shape[0] for f in feats)
    F_dim = feats[0].shape[1]
    x = torch.zeros(len(batch), T_max, F_dim)
    for i, f in enumerate(feats):
        x[i, :f.shape[0]] = f
    y = torch.tensor(labels).long()
    return x, y


def main(cfg_path: str):
    with open(cfg_path) as f:
        C = yaml.safe_load(f)

    set_seed(C["SEED"])
    out_dir = C["OUTPUT_DIR"]; ensure_dir(out_dir)

    # Data loader and preprocessing params
    eeg_cfg = EEGConfig(
        fs=C["EEG"]["FS"],
        bandpass=tuple(C["EEG"]["BANDPASS"]),
        notch=C["EEG"]["NOTCH"],
        epoch_len_sec=C["EEG"]["EPOCH_LEN_SEC"],
        epoch_overlap=C["EEG"]["EPOCH_OVERLAP"],
        use_ica=C["EEG"]["USE_ICA"],
    )

    loader = RepODLoader(C["DATA_DIR"], eeg_cfg, C["EEG"]["CHANNELS"])
    files = loader.list_files()
    tr_files, va_files, te_files = split_by_subject(files, C["TRAINING"]["VAL_SPLIT_SUBJECT"], C["TRAINING"]["TEST_SPLIT_SUBJECT"], C["SEED"])

    def extract_files(fs):
        X_list, y_list, subj_index = [], [], []
        for f in fs:
            X, y = loader.load_subject(f)            # X: (E, C, T)
            # compute features per epoch -> (E, C, F)
            feats = []
            for e in X:
                feats.append(extract_epoch_features(
                    e, eeg_cfg.fs, C["FEATURES"]["WELCH"]["FREQ_BANDS"],
                    wavelet=C["FEATURES"]["WAVELET"]["FAMILY"],
                    level=C["FEATURES"]["WAVELET"]["LEVELS"],
                    stats=tuple(C["FEATURES"]["WAVELET"]["STATS"]))
                )
            F = np.stack(feats)  # (E, C, F')
            # collapse channels as tokens for 1D CNN; for transformer we keep epochs as tokens
            X_list.append(F)
            y_list.append(y[0])  # subject label
            subj_index.append(os.path.basename(f))
        return X_list, np.array(y_list), subj_index

    X_tr, y_tr, subj_tr = extract_files(tr_files)
    X_va, y_va, subj_va = extract_files(va_files)
    X_te, y_te, subj_te = extract_files(te_files)

    # feature scaling across features per channel
    # reshape all epochs to 2D: (sum_E*C, F)
    def stack_all(X_lists):
        mats = []
        for F in X_lists:
            mats.append(F.reshape(-1, F.shape[-1]))
        return np.concatenate(mats, axis=0)

    scaler = StandardScaler().fit(stack_all(X_tr))

    def to_pytorch_items(X_list, y_list, model_kind):
        items = []
        for F, y in zip(X_list, y_list):
            E, C, Fdim = F.shape
            F = scaler.transform(F.reshape(-1, Fdim)).reshape(E, C, Fdim)
            if model_kind == "cnn1d":
                # collapse epochs by averaging -> sequence per channel
                x = torch.tensor(F.mean(axis=0), dtype=torch.float32)  # (C, Fdim)
                # reshape to (C, L) where L=Fdim
                x = x  # already (C, F)
                items.append((x, int(y)))
            else:
                # transformer expects tokens (epochs) with feature dim = C*F
                x = torch.tensor(F.reshape(E, C*Fdim), dtype=torch.float32)  # (E, C*F)
                items.append((x, int(y)))
        return items

    model_kind = C["MODEL"]["KIND"]
    train_items = to_pytorch_items(X_tr, y_tr, model_kind)
    val_items   = to_pytorch_items(X_va, y_va, model_kind)
    test_items  = to_pytorch_items(X_te, y_te, model_kind)

    if model_kind == "cnn1d":
        in_ch = train_items[0][0].shape[0]
        in_len = train_items[0][0].shape[1]
        model = CNN1DClassifier(in_ch, in_len, n_classes=2,
                                channels=tuple(C["MODEL"]["CNN1D"]["CHANNELS"]),
                                kernels=tuple(C["MODEL"]["CNN1D"]["KERNEL_SIZES"]),
                                dropout=C["MODEL"]["CNN1D"]["DROPOUT"]) 
        collate = lambda b: (
            torch.stack([x for x, _ in b], dim=0),
            torch.tensor([y for _, y in b], dtype=torch.long)
        )
    else:
        feat_dim = train_items[0][0].shape[-1]
        model = TransformerClassifier(
            n_tokens=None, feat_dim=feat_dim,
            d_model=C["MODEL"]["TRANSFORMER"]["D_MODEL"],
            nhead=C["MODEL"]["TRANSFORMER"]["N_HEAD"],
            nlayers=C["MODEL"]["TRANSFORMER"]["N_LAYERS"],
            dim_ff=C["MODEL"]["TRANSFORMER"]["DIM_FF"],
            dropout=C["MODEL"]["TRANSFORMER"]["DROPOUT"],
            n_classes=2,
        )
        collate = collate_pad

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader = DataLoader(EEGFeatureDataset(train_items), batch_size=C["TRAINING"]["BATCH_SIZE"], shuffle=True, num_workers=C["N_WORKERS"], collate_fn=collate)
    val_loader   = DataLoader(EEGFeatureDataset(val_items), batch_size=C["TRAINING"]["BATCH_SIZE"], shuffle=False, num_workers=C["N_WORKERS"], collate_fn=collate)
    test_loader  = DataLoader(EEGFeatureDataset(test_items), batch_size=C["TRAINING"]["BATCH_SIZE"], shuffle=False, num_workers=C["N_WORKERS"], collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=C["TRAINING"]["LR"], weight_decay=C["TRAINING"]["WEIGHT_DECAY"])
    crit = nn.CrossEntropyLoss()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=C["TRAINING"]["MIXED_PRECISION"])

    best_val = float('inf'); patience = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, C["TRAINING"]["MAX_EPOCHS"] + 1):
        model.train(); tr_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=C["TRAINING"]["MIXED_PRECISION"]):
                logits = model(xb)
                loss = crit(logits, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(opt)
            scaler_amp.update()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        # val
        model.eval(); va_loss = 0; y_true=[]; y_prob=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                va_loss += loss.item() * xb.size(0)
                prob = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()
                y_prob.extend(prob.tolist()); y_true.extend(yb.cpu().numpy().tolist())
        va_loss /= len(val_loader.dataset)
        m = compute_metrics(np.array(y_true), np.array(y_prob))

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(m["acc"])
        history["val_f1"].append(m["f1"])

        print(f"Epoch {epoch:03d}: tr_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={m['acc']:.3f} val_f1={m['f1']:.3f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss; patience = 0
            torch.save({"model": model.state_dict()}, os.path.join(out_dir, "best.pt"))
        else:
            patience += 1
            if patience >= C["TRAINING"]["EARLY_STOP_PATIENCE"]:
                print("Early stopping!"); break

    # Evaluate on test
    ckpt = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"]) ; model.eval()
    y_true=[]; y_prob=[]
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            prob = torch.softmax(model(xb), dim=-1)[:,1].cpu().numpy()
            y_prob.extend(prob.tolist()); y_true.extend(yb.cpu().numpy().tolist())
    test_metrics = compute_metrics(np.array(y_true), np.array(y_prob))

    # Save artifacts
    np.save(os.path.join(out_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(out_dir, "scaler_scale.npy"), scaler.scale_)
    meta = {"config": C, "test_metrics": test_metrics}
    save_json(meta, os.path.join(out_dir, "meta.json"))

    # Visualizations
    viz.plot_history(history, out_dir)
    if C["LOGGING"]["SAVE_CM"]:
        viz.plot_confusion_matrix(test_metrics["cm"], ["Healthy","SZ"], os.path.join(out_dir, "cm.png"))
    if C["LOGGING"]["SAVE_ROC"]:
        viz.plot_roc(np.array(y_true), np.array(y_prob), os.path.join(out_dir, "roc.png"))

    # t-SNE of subject-level features (use validation+test)
    # flatten items -> mean over tokens
    feats = [] ; labs=[]
    for items in (val_items + test_items):
        x, y = items
        feats.append(x.mean(dim=0).numpy())
        labs.append(y)
    feats = np.stack(feats)
    viz.plot_tsne(feats, np.array(labs), os.path.join(out_dir, "tsne.png"))

    # Export ONNX for FastAPI/Streamlit later
    if C["EXPORT"]["SAVE_ONNX"]:
        dummy = None
        if model_kind == "cnn1d":
            dummy = torch.randn(1, train_items[0][0].shape[0], train_items[0][0].shape[1]).to(device)
        else:
            dummy = torch.randn(1, 20, train_items[0][0].shape[1]).to(device) # 20 tokens placeholder
        torch.onnx.export(model, dummy, os.path.join(out_dir, "model.onnx"), opset_version=C["EXPORT"]["ONNX_OPSET"], input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    args = ap.parse_args()
    main(args.config)
