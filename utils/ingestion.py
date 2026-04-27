import pandas as pd

def load_csv(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df
