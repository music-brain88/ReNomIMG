import numpy as np
from renom_img.api.utility.box import *
from renom_img.api.utility.nms import *
import warnings

cpdef get_prec_and_rec(pred_list, gt_list, n_class=None, iou_threshold=0.5):
    """ This function calculates precision and recall value of provided ground truth box list(gt_list) and predicted box list(pred_list).

    Args:
        gt_list (list):
        pred_list (list): A list of predicted bounding boxes.
        n_class(int): The number of classes
        iou_threshold(float): This represents the ratio of overlapped area between two boxes. Defaults to 0.5

    Returns:
        4-tuples. Each element represents a dictionary pf precisions, a dictionary of recall. the number of predicted boxes which match to ground truth boxes,
        and the number of positive boxes for each class.
    """

    class_map = np.unique(np.concatenate([[g['name'] for gt in gt_list for g in gt], [p['name'] for pre in pred_list for p in pre]]))

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
        gt_names = [obj['name'] for obj in gt_list_per_img]

        pred_labels = [obj['class'] for obj in pred_list_per_img]
        pred_names = [obj['name'] for obj in pred_list_per_img]
        pred_boxes = [obj['box'] for obj in pred_list_per_img]
        pred_confs = [float(obj['score']) for obj in pred_list_per_img]

        for l in gt_names:
            n_pos_list[l] += 1

        gt_seen = np.zeros(len(gt_boxes), dtype=bool)
        for label, name, box, conf in zip(pred_labels, pred_names, pred_boxes, pred_confs):
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
                match[name].append(0)
                scores[name].append(conf)
                continue

            if maxiou >= iou_threshold:
                if not gt_seen[maxiou_id]:
                    match[name].append(1)
                    gt_seen[maxiou_id] = True
                else:
                    match[name].append(0)
            else:
                match[name].append(0)
            scores[name].append(conf)

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
    """ Returns AP and mAP

    Args:
        prec(dict): Dictionary of precision for each class returned by get_prec_and_rec method
        rec(dict): Dictionary of recall for each class returned by get_prec_and_rec method

    Returns:
        2-tuple. Each element represetns a dictionary of AP for each class and mAP (mean Average Precision).
    """
    class_map = list(prec.keys())
    aps = {}
    no_target_class = []
    for c in class_map:
        if prec[c] is None or rec[c] is None:
            aps[c] = 0.0
            no_target_class.append(c)
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
    if len(no_target_class) > 2:
        tmp = no_target_class[:3]
        tmp.append('...')
        warnings.warn("There is no following classes in the target data, (%s)"%",".join(tmp))
    elif len(no_target_class) > 0:
        warnings.warn("There is no following classes in the target data. (%s)"%",".join(no_target_class))
    return aps, mAP


cpdef get_mean_iou(pred_list, gt_list, n_class=None, iou_threshold=0.5, n_round_off=3):
    """ Mean IoU

    Args:
        gt_list (list):
        pred_list (list): A list of predicted bounding boxes.
        n_class(int): The number of classes
        iou_threshold(float): This represents the ratio of overlapped area between two boxes. Defaults to 0.5

    Returns:
        2-tuple. Each element represents a dictioanry of IoU for each class and mean IoU.
    """

    class_map = np.unique(np.concatenate([[g['name'] for gt in gt_list for g in gt], [p['name'] for pre in pred_list for p in pre]]))

    ious = {}
    for c in class_map:
        ious[c] = []


    for i in range(len(gt_list)):
        gt_list_per_img = gt_list[i]
        pred_list_per_img = pred_list[i]

        gt_labels = [obj['class'] for obj in gt_list_per_img]
        gt_names = [obj['name'] for obj in gt_list_per_img]
        gt_boxes = [obj['box'] for obj in gt_list_per_img]

        pred_labels = [obj['class'] for obj in pred_list_per_img]
        pred_names = [obj['name'] for obj in pred_list_per_img]
        pred_boxes = [obj['box'] for obj in pred_list_per_img]

        gt_seen = np.zeros(len(gt_boxes), dtype=bool)
        for label, box in zip(pred_labels, pred_boxes):
            x1, y1, x2, y2 = transform2xy12(box)

            maxiou = -1
            maxiou_id = -1
            for j, (gt_label, name, gt_box) in enumerate(zip(gt_labels, gt_names, gt_boxes)):
                if gt_label != label:
                    continue
                gt_x1, gt_y1, gt_x2, gt_y2 = transform2xy12(gt_box)

                iou = calc_iou_xyxy([x1, y1, x2, y2], [gt_x1, gt_y1, gt_x2, gt_y2])
                if iou > maxiou:
                    maxiou = iou
                    maxiou_id = j

            if maxiou >= iou_threshold:
                if not gt_seen[maxiou_id]:
                    ious[name].append(maxiou)
                    gt_seen[maxiou_id] = True
                else:
                    continue
            else:
                continue

    mean_iou_per_cls = {}
    no_target_class = []
    target_class = np.unique([g['name'] for gt in gt_list for g in gt])
    for k, v in ious.items():
        if len(v) > 0:
            mean_iou_per_cls[k] =  round(np.nanmean(v), n_round_off)
        else:
            mean_iou_per_cls[k] = 0.0
            if k not in target_class:
                no_target_class.append(k)
    if len(no_target_class) > 2:
        tmp = no_target_class[:3]
        tmp.append('...')
        warnings.warn("There is no following classes in the target data, (%s)"%",".join(tmp))
    elif len(no_target_class) > 0:
        warnings.warn("There is no following classes in the target data. (%s)"%",".join(no_target_class))
    mean_iou = round(np.nanmean(list(mean_iou_per_cls.values())), n_round_off)
    return mean_iou_per_cls, mean_iou

cpdef get_prec_rec_iou(pred_list, gt_list, n_class=None, iou_threshold=0.5, n_round_off=3):
    """ Returns preision, recall and IoU.

    Args:
        gt_list (list):
        pred_list (list): A list of predicted bounding boxes.
        n_class(int): The number of classes
        iou_threshold(float): This represents the ratio of overlapped area between two boxes. Defaults to 0.5

    Returns:
        4-tuple. Each element represetns a ditionary of precision foe each class, a dictionary of recall for each class,
        a dictionary for IoU for each class, and mean IoU (float).
    """

    class_map = np.unique(np.concatenate([[g['name'] for gt in gt_list for g in gt], [p['name'] for pre in pred_list for p in pre]]))

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
        gt_names = [obj['name'] for obj in gt_list_per_img]
        gt_boxes = [obj['box'] for obj in gt_list_per_img]

        pred_labels = [obj['class'] for obj in pred_list_per_img]
        pred_names = [obj['name'] for obj in pred_list_per_img]
        pred_boxes = [obj['box'] for obj in pred_list_per_img]
        pred_confs = [obj['score'] for obj in pred_list_per_img]

        for l in gt_names:
            n_pos_list[l] += 1

        gt_seen = np.zeros(len(gt_boxes), dtype=bool)
        for label, name, box, conf in zip(pred_labels, pred_names, pred_boxes, pred_confs):
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
                match[name].append(0)
                scores[name].append(conf)
                continue

            if maxiou >= iou_threshold:
                if not gt_seen[maxiou_id]:
                    match[name].append(1)
                    ious[name].append(maxiou)
                    gt_seen[maxiou_id] = True
                else:
                    match[name].append(0)
            else:
                match[name].append(0)
            scores[name].append(conf)

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

    mean_iou_per_cls = {}
    no_target_class = []
    target_class = np.unique([g['name'] for gt in gt_list for g in gt])
    for k, v in ious.items():
        if len(v) > 0:
            mean_iou_per_cls[k] =  round(np.nanmean(v), n_round_off)
        else:
            mean_iou_per_cls[k] = 0.0
            if k in target_class:
                no_target_class.append(k)
    if len(no_target_class) > 2:
        warnings.warn("There is no following classes in the target data, (%s)"%",".join(no_target_class[:3]) + '...')
    elif len(no_target_class) > 0:
        warnings.warn("There is no following classes in the target data. (%s)"%",".join(no_target_class))
    mean_iou = round(np.nanmean(list(mean_iou_per_cls.values())), n_round_off)
    return precisions, recalls, mean_iou_per_cls, mean_iou
