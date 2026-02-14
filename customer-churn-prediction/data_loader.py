import pandas as pd

def load_data(path="data/churn.csv"):
    df = pd.read_csv(path)
    return df
