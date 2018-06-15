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
    

def nms(box_list, threshold, return_type='box'):
    """
    NMS(Non maximum suppression)

    Args:
        box_list:
        threshold:
        return_type:
    """
    cdef float iou;
    result = []

    for i, box1 in enumerate(box_list):
        broken = False
        for j, box2 in enumerate(box_list[i+1:]):
            iou = calc_iou(box1, box2)
            if iou > threshold:
                broken = True
                break
        if return_type=='box' and not broken:
            result.append(box1)
        elif return_type=='index' and not broken:
            result.append(i)
    return result


def soft_nms(box_list, threshold, return_type='box'):
    """
    Soft-NMS.

    Reference:
        Navaneeth Bodla, Bharat Singh, Rama Chellappa, Larry S. Davis, 
        Soft-NMS -- Improving Object Detection With One Line of Code
        https://arxiv.org/abs/1704.04503
    """
    pass
