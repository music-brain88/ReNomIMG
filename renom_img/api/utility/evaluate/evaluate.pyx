import numpy as np
from ..box import *

def get_prec_and_rec(gt_list, pred_list, n_class=None, iou_threshold=0.5):
    """
    This function calculates precision and recall value of provided ground truth box list(gt_list) and
    predicted box list(pred_list).

    # TODO: write reference

    Args:
      gt_list (list):
      pred_list (list): A list of predicted bounding boxes.

    predict_list:
      [
            [ # Objects of 1st image.
                {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score(float)},
                {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score(float)},
                ...
            ],
            [ # Objects of 2nd image.
                {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score(float)},
                {'box': [x(float), y, w, h], 'clas': class_id(int), 'score': score(float)},
                ...
            ]
        ]
    """

    if not n_class:
        class_map = np.unique(np.concatenate([[g['class'] for gt in gt_list for g in gt], [p['class'] for pre in pred_list for p in pre]]))
        n_class = len(class_map)
    else:
        class_map = np.arange(n_class)

    n_pos_list = dict()
    match = {}
    scores = {}
    for c in class_map:
        n_pos_list[c] = 0
        match[c] = []
        scores[c] = []

    for i in range(len(gt_list)):
        gt_list_per_img = gt_list[i]
        pred_list_per_img = pred_list[i]

        gt_labels = [obj['class'] for obj in gt_list_per_img]
        gt_boxes = [obj['box'] for obj in gt_list_per_img]

        pred_labels = [obj['class'] for obj in pred_list_per_img]
        pred_boxes = [obj['box'] for obj in pred_list_per_img]
        pred_confs = [float(obj['score']) for obj in pred_list_per_img]

        for l in gt_labels:
            n_pos_list[l] += 1

        gt_seen = np.zeros(len(gt_boxes), dtype=bool)
        for label, box, conf in zip(pred_labels, pred_boxes, pred_confs):
            x1, y1, x2, y2 = transform2xy12(box)

            maxiou = -1
            maxiou_id = -1
            for j, (gt_label, gt_box) in enumerate(zip(gt_labels, gt_boxes)):
                if gt_label != label:
                    continue
                gt_x1, gt_y1, gt_x2, gt_y2 = transform2xy12(gt_box)

                iou = calc_iou_xyxy([x1, y1, x2, y2], [gt_x1, gt_y1, gt_x2, gt_y2])
                if iou > maxiou:
                    maxiou = iou
                    maxiou_id = j

            if maxiou < 0:
                match[label].append(0)
                scores[label].append(conf)
                continue

            if maxiou >= iou_threshold:
                if not gt_seen[maxiou_id]:
                    match[label].append(1)
                    gt_seen[maxiou_id] = True
                else:
                    match[label].append(0)
            else:
                match[label].append(0)
            scores[label].append(conf)

    precisions = []
    recalls = []
    for l in range(n_class):
        sorted_indices = np.argsort(scores[l])[::-1]
        match_per_cls = np.array(match[l])
        match_per_cls = match_per_cls[sorted_indices]
        tp = np.cumsum(match_per_cls == 1)
        fp = np.cumsum(match_per_cls == 0)
        precision = tp.astype(float) / (tp+fp).astype(float)
        precisions.append(precision)
        if n_pos_list[l] > 0:
            recall = tp.astype(float) / float(n_pos_list[l])
        else:
            recall = None
        recalls.append(recall)
    return precisions, recalls


def get_ap_and_map(prec, rec, n_round_off=2):
    """
    prec: List of precision for each class returned by get_prec_and_rec method
    rec: List of recall for each class returned by get_prec_and_rec method
    """
    n_class = len(prec)
    aps = {}
    for c in range(n_class):
        if prec[c] is None or rec[c] is None:
            aps[c] = np.nan
            continue
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(rec[c] >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(prec[c])[rec[c]>=t])
            ap += p / 11
        aps[c] = round(ap * 100, n_round_off)
    mAP = round(np.nanmean(np.array(list(aps.values()))), n_round_off)
    return aps, mAP


def get_mean_iou(gt_list, pred_list, n_class=None, iou_threshold=0.5):
    """
    predict_list:
    [
            [ # Objects of 1st image.
                {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score},
                {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score},
                ...
            ],
            [ # Objects of 2nd image.
                {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score},
                {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score},
                ...
            ]
        ]
    """

    if not n_class:
        class_map = np.unique(np.concatenate([[g['class'] for gt in gt_list for g in gt], [p['class'] for pre in pred_list for p in pre]]))
        n_class = len(class_map)
    else:
        class_map = np.arange(n_class)

    ious = {}
    for c in class_map:
        ious[c] = []


    for i in range(len(gt_list)):
        gt_list_per_img = gt_list[i]
        pred_list_per_img = pred_list[i]

        gt_labels = [obj['class'] for obj in gt_list_per_img]
        gt_boxes = [obj['box'] for obj in gt_list_per_img]

        pred_labels = [obj['class'] for obj in pred_list_per_img]
        pred_boxes = [obj['box'] for obj in pred_list_per_img]

        gt_seen = np.zeros(len(gt_boxes), dtype=bool)
        for label, box in zip(pred_labels, pred_boxes):
            x1, y1, x2, y2 = transform2xy12(box)

            maxiou = -1
            maxiou_id = -1
            for j, (gt_label, gt_box) in enumerate(zip(gt_labels, gt_boxes)):
                if gt_label != label:
                    continue
                gt_x1, gt_y1, gt_x2, gt_y2 = transform2xy12(gt_box)

                iou = calc_iou_xyxy([x1, y1, x2, y2], [gt_x1, gt_y1, gt_x2, gt_y2])
                if iou > maxiou:
                    maxiou = iou
                    maxiou_id = j

            if maxiou >= iou_threshold:
                if not gt_seen[maxiou_id]:
                    ious[label].append(maxiou)
                    gt_seen[maxiou_id] = True
                else:
                    continue
            else:
                continue
    mean_iou_per_cls = [np.mean(iou_list) for iou_list in ious.values()]
    mean_iou = np.nanmean(mean_iou_per_cls)
    return mean_iou_per_cls, mean_iou

def get_prec_rec_iou(gt_list, pred_list, n_class=None, iou_threshold=0.5):
    """
    predict_list:
    [
            [ # Objects of 1st image.
                {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score},
                {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score},
                ...
            ],
            [ # Objects of 2nd image.
                {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score},
                {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score},
                ...
            ]
        ]
    """

    if not n_class:
        class_map = np.unique(np.concatenate([[g['class'] for gt in gt_list for g in gt], [p['class'] for pre in pred_list for p in pre]]))
        n_class = len(class_map)
    else:
        class_map = np.arange(n_class)

    n_pos_list = dict()
    match = {}
    scores = {}
    ious = {}
    for c in class_map:
        n_pos_list[c] = 0
        match[c] = []
        scores[c] = []
        ious[c] = []

    for i in range(len(gt_list)):
        gt_list_per_img = gt_list[i]
        pred_list_per_img = pred_list[i]

        gt_labels = [obj['class'] for obj in gt_list_per_img]
        gt_boxes = [obj['box'] for obj in gt_list_per_img]

        pred_labels = [obj['class'] for obj in pred_list_per_img]
        pred_boxes = [obj['box'] for obj in pred_list_per_img]
        pred_confs = [obj['score'] for obj in pred_list_per_img]

        for l in gt_labels:
            n_pos_list[l] += 1

        gt_seen = np.zeros(len(gt_boxes), dtype=bool)
        for label, box, conf in zip(pred_labels, pred_boxes, pred_confs):
            x1, y1, x2, y2 = transform2xy12(box)

            iou_list = []
            indices = []
            maxiou = -1
            maxiou_id = -1
            for j, (gt_label, gt_box) in enumerate(zip(gt_labels, gt_boxes)):
                if gt_label != label:
                    continue
                gt_x1, gt_y1, gt_x2, gt_y2 = transform2xy12(gt_box)

                iou = calc_iou_xyxy([x1, y1, x2, y2], [gt_x1, gt_y1, gt_x2, gt_y2])
                if iou > maxiou:
                    maxiou = iou
                    maxiou_id = j

            if maxiou < 0:
                match[label].append(0)
                scores[label].append(conf)
                continue

            if maxiou >= iou_threshold:
                if not gt_seen[maxiou_id]:
                    match[label].append(1)
                    ious[label].append(maxiou)
                    gt_seen[maxiou_id] = True
                else:
                    match[label].append(0)
            else:
                match[label].append(0)
            scores[label].append(conf)

    precisions = []
    recalls = []
    for l in range(n_class):
        sorted_indices = np.argsort(scores[l])[::-1]
        match_per_cls = np.array(match[l])
        match_per_cls = match_per_cls[sorted_indices]
        tp = np.cumsum(match_per_cls == 1)
        fp = np.cumsum(match_per_cls == 0)
        precision = tp.astype(float) / (tp+fp).astype(float)
        precisions.append(precision)
        if n_pos_list[l] > 0:
            recall = tp.astype(float) / float(n_pos_list[l])
        else:
            recall = None
        recalls.append(recall)

    mean_iou_per_cls = [np.mean(iou_list) for iou_list in ious.values()]
    mean_iou = np.nanmean(mean_iou_per_cls)
    mean_iou = mean_iou if not np.isnan(mean_iou) else 0.0
    return precisions, recalls, mean_iou_per_cls, mean_iou

