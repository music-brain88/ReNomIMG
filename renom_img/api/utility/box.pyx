import numpy as np
from itertools import chain

cpdef rescale(box, before_size, after_size):
    """
    rescale(box, before_size, after_size)

    Rescale box coordinates and size to specific size.

    Args:
        box(list): This list has 4 variables that represent above coordinates.
        before_size(float): Size of the box before rescaling.
        after_size(float): Size of the box before rescaling.

    """
    for d in chain.from_iterable(box):
        d["box"] = [
            d["box"][0]/before_size[0] * after_size[0],
            d["box"][1]/before_size[1] * after_size[1],
            d["box"][2]/before_size[0] * after_size[0],
            d["box"][3]/before_size[1] * after_size[1],
        ] 
    

cpdef transform2xywh(box):
    """
    transform2xywh(box)

    This function changes box's coordinate format from (x1, y1, x2, y2) to
    (x, y, w, h).
    
    (``x1``, ``y1``) represents the coordinate of upper left corner.
    (``x2``, ``y2``) represents the coordinate of lower right corner.

    (``x``, ``y``) represents the center of bounding box.
    (``w``, ``h``) represents the width and height of bonding box.

    The format of argument box have to be following example.

    .. code-block :: python

        [x1(float), y1(float), x2(float), y2(float)]

    Args:
        box(list): This list has 4 variables that represent above coordinates.

    Return:
        (list): Returns reformatted bounding box.

    """
    cdef float x1, y1, x2, y2;
    cdef float x, y, w, h;
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x = x1 + w/2.0
    y = y1 + h/2.0
    return (x, y, w, h)

cpdef transform2xy12(box):
    """
    transform2xy12(box)

    This function changes box's coordinate format from (x, y, w, h) to
    (x1, y1, x2, y2).

    (``x``, ``y``) represents the center of bonding box.
    (``w``, ``h``) represents the width and height of bonding box.

    (``x1``, ``y1``) represents the coordinate of upper left corner.
    (``x2``, ``y2``) represents the coordinate of lower right corner.

    The format of argument box have to be following example.

    .. code-block :: python

        [x(float), y(float), w(float), h(float)]

    Args:
        box(list): This list has 4 variables that represent above coordinates.

    Return:
        (list): Returns reformatted bounding box.

    """
    cdef float x1, y1, x2, y2;
    cdef float x, y, w, h;
    x, y, w, h = box
    x1 = x - w/2.0
    y1 = y - h/2.0
    x2 = x + w/2.0
    y2 = y + h/2.0
    return (x1, y1, x2, y2)

cpdef calc_iou_xyxy(box1, box2):
    """
    calc_iou_xyxy(box1, box2)

    This function calculates IOU in the coordinate format (x1, y1, x2, y2).

    (``x1``, ``y1``) represents the coordinate of upper left corner.
    (``x2``, ``y2``) represents the coordinate of lower right corner.

    The format of argument box have to be following example.

    .. code-block :: python

        [x1(float), y1(float), x2(float), y2(float)]

    Args:
        [box1(list), box2(list)]: List of boxes. Each box has 4 variables that represent above coordinates.

    Return:
        (float): Returns value of IOU.

    """
    inter_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if inter_h <= 0 or inter_w <= 0: return 0
    inter = inter_h * inter_w
    union = (box1[2] - box1[0])*(box1[3] - box1[1]) + (box2[2] - box2[0])*(box2[3] - box2[1]) - inter
    iou = inter/union
    return iou

cpdef calc_iou_xywh(box1, box2):
    """
    calc_iou_xywh(box1, box2)

    This function calculates IOU in the coordinate format (x1, y1, x2, y2).

    (``x1``, ``y1``) represents the coordinate of upper left corner.
    (``x2``, ``y2``) represents the coordinate of lower right corner.

    The format of argument box have to be following example.

    .. code-block :: python

        [x1(float), y1(float), x2(float), y2(float)]

    Args:
        [box1(list), box2(list)]: List of boxes. Each box has 4 variables that represent above coordinates.

    Return:
        (float): Returns value of IOU.

    """
    box1 = (box1[0] - box1[2]/2.0, box1[1] - box1[3]/2.0, box1[0] + box1[2]/2.0, box1[1] + box1[3]/2.0)
    box2 = (box2[0] - box2[2]/2.0, box2[1] - box2[3]/2.0, box2[0] + box2[2]/2.0, box2[1] + box2[3]/2.0)
    inter_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if inter_h <= 0 or inter_w <= 0: return 0
    inter = inter_h * inter_w
    union = (box1[2] - box1[0])*(box1[3] - box1[1]) + (box2[2] - box2[0])*(box2[3] - box2[1]) - inter
    iou = inter/union
    return iou
