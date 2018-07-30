import numpy as np
from renom_img.api.utility.box import *
from renom_img.api.utility.nms import *

cpdef get_prec_and_rec(pred_list, gt_list, n_class=None, iou_threshold=0.5):
    """
    This function calculates precision and recall value of provided ground truth box list(gt_list) and
    predicted box list(pred_list).

    # TODO: write reference

    Args:
        gt_list (list):
        pred_list (list): A list of predicted bounding boxes.

            .. code-block :: python

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

        n_class(int): The number of classes
        iou_threshold(float): This represents the ratio of overlapped area between two boxes. Defaults to 0.5

    Returns:
        precisions(dict): Dictionary of precision for each class
        recalls(dict): Dictionary of recall for each class
        n_pred: The number of predicted objects for each class
        n_pos_list: The number of positive objects for each class

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

    precisions = {}
    recalls = {}
    n_pred = {}
    for l in class_map:
        sorted_indices = np.argsort(scores[l])[::-1]
        match_per_cls = np.array(match[l])
        match_per_cls = match_per_cls[sorted_indices]
        tp = np.cumsum(match_per_cls == 1)
        fp = np.cumsum(match_per_cls == 0)
        if len(tp) > 0:
            n_pred[l] = tp[-1]
        else:
            n_pred[l] = 0
        precision = tp.astype(float) / (tp+fp).astype(float)
        precisions[l] = precision
        if n_pos_list[l] > 0:
            recall = tp.astype(float) / float(n_pos_list[l])
        else:
            recall = None
        recalls[l] = recall
    return precisions, recalls, n_pred, n_pos_list


cpdef get_ap_and_map(prec, rec, n_class=None, n_round_off=3):
    """ Computing Ap and mAP
    Args:
        prec(dict): Dictionary of precision for each class returned by get_prec_and_rec method
        rec(dict): Dictionary of recall for each class returned by get_prec_and_rec method

    Returns:
        aps(dict): dictionary of AP for each class
        mAP(float): mean Average Precision
    """
    class_map = list(prec.keys())
    aps = {}
    for c in class_map:
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
        aps[c] = round(ap, n_round_off)
    mAP = round(np.nanmean(list(aps.values())), n_round_off)
    return aps, mAP


cpdef get_mean_iou(pred_list, gt_list, n_class=None, iou_threshold=0.5, n_round_off=3):
    """ Computing mean IoU
    Args:
        gt_list (list):
        pred_list (list): A list of predicted bounding boxes.

            .. code-block :: python

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

        n_class(int): The number of classes
        iou_threshold(float): This represents the ratio of overlapped area between two boxes. Defaults to 0.5

    Returns:
        mean_iou_per_cls(dict): Mean IoU for each class
        mean_iou: Average mean IoU in all classes
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
    mean_iou_per_cls = {k: round(np.mean(v), n_round_off) if v is not np.nan else np.nan for k, v in ious.items()}
    mean_iou = round(np.nanmean(list(mean_iou_per_cls.values())), n_round_off)
    return mean_iou_per_cls, mean_iou

cpdef get_prec_rec_iou(pred_list, gt_list, n_class=None, iou_threshold=0.5, n_round_off=3):
    """
    Args:
        gt_list (list):
        pred_list (list): A list of predicted bounding boxes.

            .. code-block :: python
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

        n_class(int): The number of classes
        iou_threshold(float): This represents the ratio of overlapped area between two boxes. Defaults to 0.5

    Returns:
        precisions(dict): Dictionary of precision for each class
        recalls(dict): Dictionary of recall for each class
        mean_iou_per_cls(dict): Mean IoU for each class
        mean_iou: Average mean IoU in all classes
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

    precisions = {}
    recalls = {}
    for l in class_map:
        sorted_indices = np.argsort(scores[l])[::-1]
        match_per_cls = np.array(match[l])
        match_per_cls = match_per_cls[sorted_indices]
        tp = np.cumsum(match_per_cls == 1)
        fp = np.cumsum(match_per_cls == 0)
        precision = tp.astype(float) / (tp+fp).astype(float)
        precisions[l] = precision
        if n_pos_list[l] > 0:
            recall = tp.astype(float) / float(n_pos_list[l])
        else:
            recall = None
        recalls[l] = recall

    mean_iou_per_cls = {k: round(np.nanmean(v), n_round_off) for k, v in ious.items()}
    mean_iou = np.nanmean(list(mean_iou_per_cls.values()))
    mean_iou = 0 if np.isnan(mean_iou) else mean_iou
    return precisions, recalls, mean_iou_per_cls, mean_iou

