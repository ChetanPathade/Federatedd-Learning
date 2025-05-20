import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import pi

# Load and preprocess data
df = pd.read_csv("attack_logs.csv")

# Group by AttackType and get the last entry for each (assuming that's the latest)
latest_df = df.groupby("attack_type").tail(1)

# Metrics to plot
metrics = ["precision", "recall", "f1_score"]
labels = latest_df["attack_type"].tolist()

# Extract metric values
values = latest_df[metrics].values

# Normalize axis to [0,1]
num_vars = len(metrics)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # complete the loop

# Plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

for i in range(len(values)):
    val = values[i].tolist()
    val += val[:1]
    ax.plot(angles, val, linewidth=2, label=labels[i])
    ax.fill(angles, val, alpha=0.2)

# Add feature labels to the chart
plt.xticks(angles[:-1], metrics)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="gray", size=8)
plt.title("Comparison of Attack Techniques", size=15, y=1.1)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()