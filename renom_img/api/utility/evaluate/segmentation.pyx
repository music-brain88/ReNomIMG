import numpy as np
from collections import defaultdict

cpdef get_segmentation_mean_iou(pred_list, gt_list, n_class=None, n_round_off=3, background_class=0):
    """Computing IoU for each class and mean IoU
    """
    class_num = np.max([pred_list, gt_list])

    tp = defaultdict(int)
    true_sum = defaultdict(int)
    pred_sum = defaultdict(int)

    for pred, gt in zip(pred_list, gt_list):
        for c in range(class_num+1):
            if c == background_class:
                continue
            pred_sum[c] += np.sum(np.where(pred==c, True, False))
            true_sum[c] += np.sum(np.where(gt==c, True, False))
            tp[c] += np.sum(np.where(pred==c, True, False) * np.where(gt==c, True, False))

    ious = {}
    total_area = 0
    total_tp = 0
    for c in range(class_num+1):
        if c == background_class:
            continue
        area = true_sum[c] + pred_sum[c] - tp[c]
        total_area += float(area)
        total_tp += float(tp[c])
        if area == 0:
            ious[c] == 0.
        else:
            ious[c] = float(tp[c]) / float(area)

    mean_iou = total_tp / total_area
    return ious, mean_iou
