from src.preprocess import load_data, preprocess_data, split_data
from src.feature_engineering import add_credit_features
from src.model import train_logistic_regression, train_lightgbm, evaluate_model

# Load data
df = load_data('data/loan_data.csv')

# Feature engineering
df = add_credit_features(df)

# Preprocess and split
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Train models
lr_model = train_logistic_regression(X_train, y_train)
lgb_model = train_lightgbm(X_train, y_train)

# Evaluate
print("Logistic Regression Performance:")
evaluate_model(lr_model, X_test, y_test)

print("LightGBM Performance:")
evaluate_model(lgb_model, X_test, y_test)
