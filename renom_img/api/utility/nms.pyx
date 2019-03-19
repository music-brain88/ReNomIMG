import numpy as np
from renom_img.api.utility.box import calc_iou_xyxy, calc_iou_xywh

def nms(preds, threshold=0.5):
    """ Non-Maximum Suppression

    Args:
        preds(list): A list of predicted boxes. The format is as follows.
        threshold(float, optional): Defaults to `0.5`. This represents the ratio of overlap between two boxes.


    .. code-block :: python
        :caption: **Example of the argument "preds".**

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


    Returns:
        (list): Returns reformatted bounding box.

    .. code-block :: python
        :caption: **Example of return value.**

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
    cdef int last;
    cdef float iou;
    result = []
    for pred in preds:
        scores = [obj['score'] for obj in pred]
        index = np.argsort(scores).tolist()
        tmp = []
        while len(index) > 0:
            last = index.pop()
            box1 = pred[last]
            class_id1 = box1["class"]
            tmp.append(box1)
            for j in index[:]:
                box2 = pred[j]
                class_id2 = box2["class"]
                iou = calc_iou_xywh(box1["box"], box2["box"])
                if class_id1 == class_id2 and iou > threshold:
                    index.remove(j)

        result.append(tmp)
    return result

def soft_nms(preds, threshold=0.5):
    """ Soft Non-Maximum Suppression

    Args:
        preds(list): A list of predicted boxes. The format is as follows.
        threshold(float, optional): Defaults to `0.5`. This represents the ratio of overlap between two boxes.

    .. code-block :: python
        :caption: **Example of the argument, "preds".**

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


    Returns:
        (list): Returns reformatted bounding box.

    .. code-block :: python
        :caption: **Example of the output.**

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

    References:
        | Navaneeth Bodla, Bharat Singh, Rama Chellappa, Larry S. Davis,
        | Soft-NMS -- Improving Object Detection With One Line of Code
        | https://arxiv.org/abs/1704.04503

    """

    cdef float iou;
    result = []
    for pred in preds:
        boxes = [obj['box'] for obj in pred]
        scores = [obj['score'] for obj in pred]
        class_id = [obj['class'] for obj in pred]
        tmp = []
        index = np.argsort(scores).tolist()
        while len(index) > 0:
            last = len(index) - 1
            i = index[last]
            box1 = boxes[i]
            score = scores[i]

            tmp.append(pred[i])

            index.pop(last)
            boxes.pop(i)
            scores.pop(i)

            for j, box2 in enumerate(boxes):
                iou = calc_iou_xyxy(box1, box2)
                if iou > threshold:
                    scores[j] *= (1-iou)
            index = np.argsort(scores).tolist()
        result.append(tmp)
    return result

