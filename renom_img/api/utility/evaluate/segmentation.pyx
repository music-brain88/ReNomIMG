import numpy as np
from collections import defaultdict

cpdef get_segmentation_metrics(pred_list, gt_list, n_class, round_off=3, ignore_class=0):
    """Computing IoU for each class and mean IoU
    """
    if isinstance(ignore_class, int):
        ignore_class = [ignore_class]
    assert len(pred_list)==len(gt_list)
    class_num = n_class

    tp = defaultdict(int)
    true_sum = defaultdict(int)
    pred_sum = defaultdict(int)

    #for pred, gt in zip(pred_list, gt_list):
    for i in range(len(pred_list)):
        pred = pred_list[i]
        gt = gt_list[i]
        for c in range(class_num):
            if c in ignore_class:
                continue
            pred_sum[c] += np.sum(np.where(pred==c, True, False))
            true_sum[c] += np.sum(np.where(gt==c, True, False))
            tp[c] += np.sum(np.where(pred==c, True, False) * np.where(gt==c, True, False))

    ious = {}
    precision = {}
    f1 = {}
    recall = {}
    total_area = 0
    total_tp = 0
    mean_f1 = 0
    mean_precision = 0
    mean_recall = 0
    for c in range(class_num):
        if c in ignore_class:
            continue
        area = true_sum[c] + pred_sum[c] - tp[c]
        total_area += float(area)
        total_tp += float(tp[c])
        if area == 0:
            ious[c] = 0.
        else:
            ious[c] = float(tp[c]) / float(area)

        if pred_sum[c] == 0:
            precision[c] = 0
        else:
            precision[c] = float(tp[c]) / float(pred_sum[c])

        if true_sum[c] == 0:
            recall[c] = 0
        else:
            recall[c] = float(tp[c]) / float(true_sum[c])

        if recall[c] + precision[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        else:
            f1[c] = 0.

        mean_f1 += f1[c] * (float(true_sum[c]) / np.fromiter(true_sum.values(), dtype=float).sum())
        mean_precision += precision[c] * (float(true_sum[c]) / np.fromiter(true_sum.values(), dtype=float).sum())
        mean_recall += recall[c] * (float(true_sum[c]) / np.fromiter(true_sum.values(), dtype=float).sum())

    mean_iou = total_tp / total_area
    return precision, mean_precision, recall, mean_recall, f1, mean_f1, ious, mean_iou, tp, true_sum

cpdef segmentation_iou(pred_list, gt_list, n_class=None, round_off=3, ignore_class=0):
    _, _, _, _, _, _, iou, mean_iou, _, _ = get_segmentation_metrics(pred_list, gt_list, n_class, round_off, ignore_class)
    return iou, mean_iou

cpdef segmentation_precision(pred_list, gt_list, n_class=None, round_off=3, ignore_class=0):
    precision, mean_precision, _, _, _, _, _, _, _, _ = get_segmentation_metrics(pred_list, gt_list, n_class, round_off, ignore_class)
    return precision, mean_precision

cpdef segmentation_recall(pred_list, gt_list, n_class=None, round_off=3, ignore_class=0):
    _, _, recall, mean_recall, _, _, _, _, _, _ = get_segmentation_metrics(pred_list, gt_list, n_class, round_off, ignore_class)
    return recall, mean_recall

cpdef segmentation_f1(pred_list, gt_list, n_class=None, round_off=3, ignore_class=0):
    _, _, _, _, f1, mean_f1, _, _, _, _ = get_segmentation_metrics(pred_list, gt_list, n_class, round_off, ignore_class)
    return f1, mean_f1


