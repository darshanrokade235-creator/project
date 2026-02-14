from sklearn.metrics import classification_report, roc_auc_score

def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        # ROC-AUC
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_prob)
        except:
            roc = "Not Available"

        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "roc_auc": roc,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"]
        }

    return results
