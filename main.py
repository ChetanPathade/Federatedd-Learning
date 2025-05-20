import multiprocessing
import time
import flwr as fl
from client import GenomicClient
from server import start_server
from attack import run_mia
from log_results import log_results
from gradient_attack import run_gradient_attack
from plot_results import plot_results
import numpy as np
from label_inference_attack import run_label_inference_attack
import subprocess


def start_client(client_id):
    client = GenomicClient(client_id)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)


if __name__ == "__main__":
    # Start the Flower server
    server_process = multiprocessing.Process(target=start_server)
    server_process.start()

    # Let the server start
    time.sleep(2)

    # Start clients
    client_processes = []
    for i in range(5):
        p = multiprocessing.Process(target=start_client, args=(i,))
        p.start()
        client_processes.append(p)

    # Wait for clients to finish
    for p in client_processes:
        p.join()

    # Simulate post-training analysis on one client
    print("\nRunning Membership Inference Attack (Client 0)...")
    test_client = GenomicClient(client_id=0)
    test_client.fit(None, {})         # simulate training on client 0
    test_client.evaluate(None, {})    # simulate predictions on test data

    mia_results = run_mia(test_client.train_membership_log)
    print("Membership Inference Attack Results:")
    print(mia_results)

    log_results(
        "attack_logs.csv", 
        attack_type="MIA", 
        client_count=5, 
        threshold="N/A", 
        precision=mia_results["precision"], 
        recall=mia_results["recall"], 
        f1_score=mia_results["f1_score"]
    )

    print("\nRunning Gradient-Based Membership Inference Attack...")
    threshold = 0.4
    attack_results = run_gradient_attack(test_client.per_sample_gradients, threshold=threshold, plot_roc=True)
    print("Gradient-Based MIA Results:")
    print(attack_results)

    log_results(
        "attack_logs.csv",
        "Gradient MIA",
        5,
        threshold,
        attack_results["precision"],
        attack_results["recall"],
        attack_results["f1_score"]
    )

    print("\nRunning Label Inference Attack...")
    lia_results = run_label_inference_attack(test_client.X_test, test_client.y_test)
    print("Label Inference Attack Results:")
    print(lia_results)

    log_results(
        "attack_logs.csv",
        "Label Inference Attack",
        5,
        "N/A",
        lia_results["precision"],
        lia_results["recall"],
        lia_results["f1_score"]
    )

    plot_results("attack_logs.csv")

    subprocess.run(["python", "plot_radar_chart.py"])
    # Clean shutdown
    server_process.join()
    print("\n✅ All processes completed. Results saved to attack_logs.csv")
