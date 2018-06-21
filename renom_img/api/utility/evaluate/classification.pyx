import numpy as np
from collections import defaultdict

cpdef precision_score(y_pred, y_true):
    """
    Precision score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]
    """
    p, _ = precision_recall_score(y_pred, y_true)
    return p

cpdef recall_score(y_pred, y_true):
    """
    Recall score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]
    """
    _, r = precision_recall_score(y_pred, y_true)
    return r

cpdef precision_recall_score(y_pred, y_true):
    """
    Recall score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]
    """
    tp = defaultdict(int)
    pred_sum = defaultdict(int)
    true_sum = defaultdict(int)

    for p, t in zip(y_pred, y_true):
        pred_sum[p] += 1
        true_sum[t] += 1
        if p == t:
            tp[p] += 1

    class_ids = np.unique(np.concatenate([pred_sum.keys(), true_sum.keys()]))
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

    return precision, recall

