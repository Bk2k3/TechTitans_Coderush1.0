# src/common/metrics.py
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = None
    out["cm"] = confusion_matrix(y_true, y_pred).tolist()
    out["report"] = classification_report(y_true, y_pred, output_dict=True)
    return out
