import numpy as np

cpdef transform2xywh(box):
    cdef float x1, y1, x2, y2;
    cdef float x, y, w, h;
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x = x1 + w/2.0
    y = y1 + h/2.0
    return (x, y, w, h)

cpdef transform2xy12(box):
    cdef float x1, y1, x2, y2;
    cdef float x, y, w, h;
    x, y, w, h = box
    x1 = x - w/2.0
    y1 = y - h/2.0
    x2 = x + w/2.0
    y2 = y + h/2.0
    return (x1, y1, x2, y2)

cpdef calc_iou_xyxy(box1, box2):
    inter_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if inter_h <= 0 or inter_w <= 0: return 0
    inter = inter_h * inter_w
    union = (box1[2] - box1[0])*(box1[3] - box1[1]) + (box2[2] - box2[0])*(box2[3] - box2[1]) - inter
    iou = inter/union
    return iou

cpdef calc_iou_xywh(box1, box2):
    box1 = (box1[0] - box1[2]/2.0, box1[1] - box1[3]/2.0, box1[0] + box1[2]/2.0, box1[1] + box1[3]/2.0)
    box2 = (box2[0] - box2[2]/2.0, box2[1] - box2[3]/2.0, box2[0] + box2[2]/2.0, box2[1] + box2[3]/2.0)
    inter_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if inter_h <= 0 or inter_w <= 0: return 0
    inter = inter_h * inter_w
    union = (box1[2] - box1[0])*(box1[3] - box1[1]) + (box2[2] - box2[0])*(box2[3] - box2[1]) - inter
    iou = inter/union
    return iou
