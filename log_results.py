import csv
import os

def log_results(log_file, attack_type, client_count, threshold, precision, recall, f1_score):
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "attack_type", "client_count", "threshold", "precision", "recall", "f1_score"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "attack_type": attack_type,
            "client_count": client_count,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        })