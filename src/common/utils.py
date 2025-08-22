# src/common/utils.py
import os, random, json, math
import numpy as np
import torch


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def subject_from_filename(fname: str):
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)
    return name.lower()


def label_from_filename(fname: str):
    # RepOD files use prefixes like h01.edf for healthy and s01.edf for schizophrenia
    base = os.path.basename(fname).lower()
    if base.startswith("h"):
        return 0  # healthy control
    elif base.startswith("s"):
        return 1  # schizophrenia
    # fallback: try to infer
    raise ValueError(f"Cannot infer label from filename: {fname}")


def split_by_subject(files, val_ratio=0.2, test_ratio=0.2, seed=1337):
    rng = np.random.default_rng(seed)
    subjects = sorted(list({subject_from_filename(f) for f in files}))
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = int(round(n * test_ratio))
    n_val = int(round((n - n_test) * val_ratio))
    test_subj = set(subjects[:n_test])
    val_subj = set(subjects[n_test:n_test + n_val])
    train_subj = set(subjects[n_test + n_val:])

    def _assign(fs, subs):
        return [f for f in fs if subject_from_filename(f) in subs]

    return _assign(files, train_subj), _assign(files, val_subj), _assign(files, test_subj)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
