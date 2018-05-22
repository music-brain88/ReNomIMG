
def transform2xywh(box):
    cdef float x1, y1, x2, y2;
    cdef float x, y, w, h;
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x = x1 + w/2.0
    y = y1 + h/2.0
    return (x, y, w, h)

def transform2xy12(box):
    cdef float x1, y1, x2, y2;
    cdef float x, y, w, h;
    x, y, w, h = box
    x1 = x - w/2.0
    y1 = y - h/2.0
    x2 = x + w/2.0
    y2 = y + h/2.0
    return (x1, y1, x2, y2)
    

def calc_iou(box1, box2):
    cdef float x1b1, y1b1, x2b1, y2b1;
    cdef float x1b2, y1b2, x2b2, y2b2;
    cdef float intersection;
    cdef float union;
    
    x1b1, y1b1, x2b1, y2b1 = box1
    x1b2, y1b2, x2b2, y2b2 = box2
    intersection = (max(x1b1, x1b2) - min(x2b1, x2b2)) * \
                    (max(y1b1, y1b2) - min(y2b1, y2b2)) 
    union = (x2b1 - x1b1) * (y2b1 - y1b1) + (x2b2 - x1b2) * (y2b2 - y1b2)
    iou = intersection/(union - intersection)
    return iou    

def nms(box_list, threshold, return_type='box'):
    """
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

