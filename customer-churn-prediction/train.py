from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def train_models(X_train, y_train):
    # SMOTE for imbalanced data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_resampled, y_resampled)
        trained_models[name] = model

    return trained_models
