from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, target_column=None):
    df = df.copy()

    # Auto-detect churn column
    if target_column is None:
        possible_targets = ["churn", "Churn", "Exited", "exit", "target", "Target"]
        for col in possible_targets:
            if col in df.columns:
                target_column = col
                break

    if target_column is None:
        raise ValueError(
            f"❌ Target column not found! Available columns are: {list(df.columns)}"
        )

    print(f"✅ Target Column Detected: {target_column}")

    # Drop customer_id if exists
    if "customer_id" in df.columns:
        df.drop("customer_id", axis=1, inplace=True)

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler