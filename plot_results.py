import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_results(csv_path):
    df = pd.read_csv(csv_path)

    metrics = ["precision", "recall", "f1_score"]
    attack_types = df["attack_type"].unique()
    bar_width = 0.25
    index = np.arange(len(metrics))

    plt.figure(figsize=(10, 6))

    for i, attack in enumerate(attack_types):
        scores = df[df["attack_type"] == attack][metrics].mean().values
        plt.bar(index + i * bar_width, scores, width=bar_width, label=attack)

    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Attack Performance Comparison")
    plt.xticks(index + bar_width, metrics)
    plt.ylim(0, 1)
    plt.legend(title="Attack Type")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()