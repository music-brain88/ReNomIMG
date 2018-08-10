import numpy as np
from collections import defaultdict

cpdef precision_score(y_pred, y_true):
    """ Precision score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        Two values are returned. First output is dictionary of precision for each class.
        The second represents the mean precision for all classes.
    """
    p, mean_p, _, _, _, _ = precision_recall_f1_score(y_pred, y_true)
    return p, mean_p

cpdef recall_score(y_pred, y_true):
    """ Recall score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        Two values are returned. First output is dictionary of recall for each class.
        The second represents the mean recall for all classes.
    """
    _, _, r, mean_r, _, _ = precision_recall_f1_score(y_pred, y_true)
    return r, mean_r

cpdef f1_score(y_pred, y_true):
    """ F1 score for classification

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        Two values are returned. First output is dictionary of F1 score for each class.
        The second represents the mean F1 score for all classes.
    """
    _, _, _, _, f1_score, mean_f1_score = precision_recall_f1_score(y_pred, y_true)
    return f1_score, mean_f1_score


cpdef precision_recall_f1_score(y_pred, y_true):
    """ Returns precision, recall, F1 score

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        6-tuple. Each element represents a dictionary of precision, mean precision of float, a dictionary of recall, mean recall of float,
        a dictionary of F1 score, and F1 score of float value.
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
    """ Accuracy

    Args:
        y_pred: [class_id(int), class_id(int), ...]
        y_true: [class_id(int), class_id(int), ...]

    Return:
        Outputs accuracy whose type is float.
    """
    accuracy = np.sum(y_pred==y_true) / len(y_true)
    return accuracy

