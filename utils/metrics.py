from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = [1 if p > threshold else 0 for p in y_probs]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    return {"acc": acc, "f1": f1, "auc": auc}
