import pandas as pd
from sklearn.model_selection import train_test_split

def load_client_data(client_id, path="dataset/client_data_20000_cleaned.csv"):
    df = pd.read_csv(path)
    client_df = df[df["client_id"] == client_id]
    X = client_df.drop(["label", "client_id"], axis=1)
    y = client_df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42)