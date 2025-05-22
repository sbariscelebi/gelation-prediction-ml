import numpy as np
from sklearn.metrics import f1_score

def find_optimal_threshold(model, X_data, y_true):
    y_pred_proba = model.predict(X_data).ravel()
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx], thresholds, f1_scores
