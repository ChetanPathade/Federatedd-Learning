import csv

# Define the correct header columns
header = ["attack_type", "client_count", "threshold", "precision", "recall", "f1_score"]

# Create or overwrite the file with the header
with open("attack_logs.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()

print("✅ attack_logs.csv has been reset with the correct headers.")