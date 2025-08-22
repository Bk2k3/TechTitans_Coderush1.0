# src/common/viz.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.manifold import TSNE


def plot_history(log, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = np.arange(1, len(log["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, log["train_loss"], label="train loss")
    plt.plot(epochs, log["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200); plt.close()

    plt.figure()
    plt.plot(epochs, log["val_acc"], label="val acc")
    plt.plot(epochs, log["val_f1"], label="val f1")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_scores.png"), dpi=200); plt.close()


def plot_confusion_matrix(cm, class_names, out_path):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=class_names)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close()


def plot_roc(y_true, y_prob, out_path):
    try:
        RocCurveDisplay.from_predictions(y_true, y_prob)
        plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close()
    except Exception:
        pass


def plot_tsne(features, labels, out_path):
    z = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30).fit_transform(features)
    plt.figure()
    sns.scatterplot(x=z[:,0], y=z[:,1], hue=labels, palette="Set1", s=18)
    plt.legend(title="class"); plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close()
