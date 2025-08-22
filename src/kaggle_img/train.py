# src/kaggle_img/train.py
import os, yaml
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
from ..common.utils import set_seed, ensure_dir, save_json
from ..common.metrics import compute_metrics
from ..common import viz
from .dataset import SpectrogramFolder
from .model_cnn2d import get_backbone


def main(cfg_path: str):
    with open(cfg_path) as f:
        C = yaml.safe_load(f)

    set_seed(C["SEED"])
    out_dir = C["OUTPUT_DIR"]; ensure_dir(out_dir)

    tf_train = T.Compose([
        T.Resize((C["IMG"]["SIZE"], C["IMG"]["SIZE"])),
        T.RandomHorizontalFlip(C["IMG"]["AUG"]["HFLIP"]),
        T.RandomRotation(C["IMG"]["AUG"]["ROTATE"]),
        T.ColorJitter(*C["IMG"]["AUG"]["COLORJITTER"]),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tf_val = T.Compose([
        T.Resize((C["IMG"]["SIZE"], C["IMG"]["SIZE"])),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # If DATA_DIR has train/ and val/ use them; else split
    train_root = os.path.join(C["DATA_DIR"], 'train')
    val_root = os.path.join(C["DATA_DIR"], 'val')
    if os.path.isdir(train_root) and os.path.isdir(val_root):
        ds_train = SpectrogramFolder(train_root, transform=tf_train)
        ds_val = SpectrogramFolder(val_root, transform=tf_val)
    else:
        ds_full = SpectrogramFolder(C["DATA_DIR"], transform=tf_train)
        n_val = int(len(ds_full) * C["TRAINING"]["VAL_SPLIT"]) ; n_train = len(ds_full) - n_val
        ds_train, ds_val = random_split(ds_full, [n_train, n_val])
        ds_val.dataset.transform = tf_val  # ensure val transform

    model = get_backbone(C["MODEL"]["BACKBONE"], n_classes=2, dropout=C["MODEL"]["DROPOUT"], pretrained=C["MODEL"]["PRETRAINED"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader = DataLoader(ds_train, batch_size=C["TRAINING"]["BATCH_SIZE"], shuffle=True, num_workers=C["N_WORKERS"]) 
    val_loader   = DataLoader(ds_val, batch_size=C["TRAINING"]["BATCH_SIZE"], shuffle=False, num_workers=C["N_WORKERS"]) 

    opt = torch.optim.AdamW(model.parameters(), lr=C["TRAINING"]["LR"], weight_decay=C["TRAINING"]["WEIGHT_DECAY"])
    crit = nn.CrossEntropyLoss()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=C["TRAINING"]["MIXED_PRECISION"]) 

    best_val = float('inf'); patience = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, C["TRAINING"]["MAX_EPOCHS"] + 1):
        model.train(); tr_loss=0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), torch.tensor(yb).to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=C["TRAINING"]["MIXED_PRECISION"]):
                logits = model(xb)
                loss = crit(logits, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(opt); scaler_amp.update()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval(); va_loss=0; y_true=[]; y_prob=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), torch.tensor(yb).to(device)
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

    # Evaluate and save
    ckpt = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"]) ; model.eval()

    # Full-val metrics and visualizations
    y_true=[]; y_prob=[]
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()
            y_prob.extend(prob.tolist()); y_true.extend(np.array(yb))
    test_metrics = compute_metrics(np.array(y_true), np.array(y_prob))

    # Save meta
    save_json({"config": C, "val_metrics": test_metrics}, os.path.join(out_dir, "meta.json"))

    # Plots
    viz.plot_history(history, out_dir)
    if C["LOGGING"]["SAVE_CM"]:
        viz.plot_confusion_matrix(test_metrics["cm"], ["Healthy","SZ"], os.path.join(out_dir, "cm.png"))
    if C["LOGGING"]["SAVE_ROC"]:
        viz.plot_roc(np.array(y_true), np.array(y_prob), os.path.join(out_dir, "roc.png"))

    # Grad-CAM for a few images (ResNet only for simplicity)
    if C["LOGGING"]["SAVE_GRADCAM"] and C["MODEL"]["BACKBONE"].lower() == 'resnet18':
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            from pytorch_grad_cam.utils.image import show_cam_on_image
            import cv2
            target_layers = [model.layer4[-1]]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
            os.makedirs(os.path.join(out_dir, "gradcam"), exist_ok=True)
            count = 0
            for i in range(min(12, len(ds_val))):
                img, label = ds_val[i]
                input_tensor = img.unsqueeze(0).to(device)
                grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])
                grayscale_cam = grayscale_cam[0, :]
                rgb = img.permute(1,2,0).cpu().numpy()
                rgb = (rgb * [0.229,0.224,0.225] + [0.485,0.456,0.406]).clip(0,1)
                cam_img = show_cam_on_image(rgb.astype(np.float32), grayscale_cam, use_rgb=True)
                cv2.imwrite(os.path.join(out_dir, "gradcam", f"val_{i}_label{label}.png"), cam_img[:, :, ::-1])
                count += 1
        except Exception as e:
            print("GradCAM skipped:", e)

    # Export ONNX
    if C["EXPORT"]["SAVE_ONNX"]:
        dummy = torch.randn(1, 3, C["IMG"]["SIZE"], C["IMG"]["SIZE"]).to(device)
        torch.onnx.export(model, dummy, os.path.join(out_dir, "model.onnx"), opset_version=C["EXPORT"]["ONNX_OPSET"], input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    args = ap.parse_args()
    main(args.config)
