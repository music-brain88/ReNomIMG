import numpy as np
from collections import defaultdict

cpdef precision_score(y_pred, y_true):
    """
    precision_score(y_pred, y_true)

    Precision score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        precision(float):
    """
    p, mean_p, _, _ = precision_recall_score(y_pred, y_true)
    return p, mean_p

cpdef recall_score(y_pred, y_true):
    """
    recall_score(y_pred, y_true)

    Recall score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        recall(float):
    """
    _, _, r, mean_r = precision_recall_score(y_pred, y_true)
    return r, mean_r

cpdef precision_recall_score(y_pred, y_true):
    """
    precision_recall_score(y_pred, y_true)

    Recall score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        precision(float), recall(float)
    """
    tp = defaultdict(int)
    pred_sum = defaultdict(int)
    true_sum = defaultdict(int)

    for p, t in zip(y_pred, y_true):
        pred_sum[p] += 1
        true_sum[t] += 1
        if p == t:
            tp[p] += 1

    class_ids = set(pred_sum.keys()) | set(true_sum.keys())
    precision = {}
    recall = {}
    for c in class_ids:
        if pred_sum[c] > 0:
            precision[c] = float(tp[c]) / pred_sum[c]
        else:
            precision[c] = 0
        if true_sum[c] > 0:
            recall[c] = float(tp[c]) / true_sum[c]
        else:
            recall[c] = None
    mean_precision = np.fromiter(tp.values(), dtype=float).sum() / np.fromiter(pred_sum.values(), dtype=float).sum()
    mean_recall = np.fromiter(tp.values(), dtype=float).sum() / np.fromiter(true_sum.values(), dtype=float()).sum()
    return precision, mean_precision, recall, mean_recall

cpdef accuracy_score(y_pred, y_true):
    """
    accuracy_score(y_pred, y_true)

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        accuracy(float):
    """
    accuracy = np.sum(y_pred==y_true)
    return accuracy

