# gradient_attack.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def run_gradient_attack(per_sample_gradients, threshold=0.5, plot_roc=False):
    y_true = []
    gradient_norms = []

    for entry in per_sample_gradients:
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            gradient = entry[0]
            is_member = entry[1]
        elif isinstance(entry, dict) and "gradient" in entry and "is_member" in entry:
            gradient = entry["gradient"]
            is_member = entry["is_member"]
        else:
            raise ValueError("Unexpected format for gradient entry")

        grad_norm = np.linalg.norm(gradient)
        gradient_norms.append(grad_norm)
        y_true.append(int(is_member))

    # 🔧 Convert norms to binary predictions
    y_pred = [int(norm > threshold) for norm in gradient_norms]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if plot_roc:
        fpr, tpr, _ = roc_curve(y_true, gradient_norms)
        auc_score = roc_auc_score(y_true, gradient_norms)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Gradient-Based MIA ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("gradient_attack_roc.png")
        plt.close()

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "threshold_used": threshold,
    }