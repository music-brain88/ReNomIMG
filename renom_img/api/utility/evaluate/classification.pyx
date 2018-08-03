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
        precision(dict): {class_id: precision(float), ..}
        mean_precision(float):
    """
    p, mean_p, _, _, _, _ = precision_recall_f1_score(y_pred, y_true)
    return p, mean_p

cpdef recall_score(y_pred, y_true):
    """
    recall_score(y_pred, y_true)

    Recall score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        recall(dict): {class_id: recall(float), ..}
        mean_recall(float):
    """
    _, _, r, mean_r, _, _ = precision_recall_f1_score(y_pred, y_true)
    return r, mean_r

cpdef f1_score(y_pred, y_true):
    """
    f1_score(y_pred, y_true)

    F1 score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        f1_score(dict): {class_id: f1_score(float), ..}
        mean_f1_score(float):
    """
    _, _, _, _, f1_score, mean_f1_score = precision_recall_f1_score(y_pred, y_true)
    return f1_score, mean_f1_score


cpdef precision_recall_f1_score(y_pred, y_true):
    """
    precision_recall_score(y_pred, y_true)

    Recall score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        precision(float), recall(float)
    """
    np.bin
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
    f1 = {}
    mean_f1 = 0
    mean_precision = 0
    mean_recall = 0
    for c in class_ids:
        if pred_sum[c] > 0:
            precision[c] = float(tp[c]) / float(pred_sum[c])
        else:
            precision[c] = 0
        if true_sum[c] > 0:
            recall[c] = float(tp[c]) / float(true_sum[c])
        else:
            recall[c] = 0

        if recall[c] + precision[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        else:
            f1[c] = 0.
        mean_f1 += f1[c] * (float(true_sum[c]) / np.fromiter(true_sum.values(), dtype=float).sum())
        mean_precision += precision[c] * (float(true_sum[c]) / np.fromiter(true_sum.values(), dtype=float).sum())
        mean_recall += recall[c] * (float(true_sum[c]) / np.fromiter(true_sum.values(), dtype=float).sum())

    return precision, mean_precision, recall, mean_recall, f1, mean_f1

cpdef accuracy_score(y_pred, y_true):
    """
    accuracy_score(y_pred, y_true)

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        accuracy(float):
    """
    accuracy = np.sum(y_pred==y_true) / len(y_true)
    return accuracy

