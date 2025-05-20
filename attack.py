# attack.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def run_mia(membership_log):
    true_labels = []
    confidence_scores = []

    for _, (features, prob, is_member) in membership_log.items():
        confidence_scores.append(prob)
        true_labels.append(is_member)

    threshold = 0.5
    preds = [1 if p > threshold else 0 for p in confidence_scores]

    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }