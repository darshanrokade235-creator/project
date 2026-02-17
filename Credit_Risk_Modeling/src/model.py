from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(n_estimators=200, learning_rate=0.05)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC Score: {auc:.4f}")
    return auc
