import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Drop duplicates and missing values
    df = df.drop_duplicates()
    df = df.dropna()

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Split features and target
    X = df.drop('defaulted', axis=1)
    y = df['defaulted']

    # Standardize numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
