import os
import joblib
from data_loader import load_data
from preprocess import preprocess_data
from train import train_models
from evaluate import evaluate_models


def main():
    # Create models folder automatically
    os.makedirs("models", exist_ok=True)

    print("📌 Loading dataset...")
    df = load_data("churn.csv")

    print("📌 Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("📌 Training models with SMOTE...")
    models = train_models(X_train, y_train)

    print("📌 Evaluating models...")
    results = evaluate_models(models, X_test, y_test)

    # Print results
    for model_name, metrics in results.items():
        print(f"\n✅ Model: {model_name}")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    # Save Logistic Regression model (default)
    best_model = models["Logistic Regression"]

    joblib.dump(best_model, "models/churn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\n🎉 Model + Scaler saved successfully in models/ folder!")


if __name__ == "__main__":
 main()