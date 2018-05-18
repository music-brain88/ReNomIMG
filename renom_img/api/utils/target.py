#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np


"""Naming Rule.
- build_target_[algorithm name]

This type of function performs following transformation.
  annotation_list => algorithm specific target data

For confirming the format of annotation_list, 
see the function 'parse_xml_detection' written in load.py.
"""

def build_target_yolo(annotation_list, cells, img_size, class_mapping=None, class_num=None):
    """Use to transform a list of objects per image into a image*cells*cells*(5+classes) matrix.

    Returns:
        (ndarray): Yolo formatted target array.

    This method returns yolo formatted target array which shape is (N, cell*cell*(5 + classes_num).
    N is batch size. The array consists of following data.

    1. existence flag: A flag which indicates if an object exists in the cell.
    2. Coordinates and size(x, y, w, h): Coordinate and size of each objects.
    3. One hot represented class: The object's class number.

    [
        [existence x y w h 0 1 0 0 existence x y w h 1 0 0 0 ... ],
        [existence x y w h 0 0 1 0 existence x y w h 0 0 0 1 ... ],
    ]
    """
    # Check the class mapping.
    if class_mapping is None:
        class_dict = {}
        for annotation in annotation_list:
            for obj in annotation:
                class_dict[obj['name']] = 1
        class_mapping = {k:i for i, k in enumerate(sorted(class_dict.keys()))} 
    else:
        assert isinstance(class_mapping, dict)

    # Check the total class number is not less than given mapping.
    if class_num is not None:
        assert len(class_mapping) <= class_num
        assert max(class_mapping.values()) < class_num
    else:
        class_num = len(class_mapping)

    # Cell can be tuple, list or int.
    if isinstance(cells, (tuple, list)):
        cell_w = cells[0]
        cell_h = cells[1]
    elif isinstance(cells, int):
        cell_w = cells
        cell_h = cells

    img_w, img_h = img_size
    target = np.zeros((len(annotation_list), cell_h, cell_w, 5+class_num))
    for ind_img in range(len(target)):
        annotation = annotation_list[ind_img]
        for ind_obj in range(len(annotation)):
            obj = annotation[ind_obj]
            class_id = class_mapping[obj['name']]
            one_hot_class_id = [0]*class_id + [1] + [0]*(class_num - class_id - 1)
            truth_x = np.clip(obj['box'][0], 0, img_w)
            truth_y = np.clip(obj['box'][1], 0, img_h)
            truth_w = np.clip(obj['box'][2], 0, img_w)
            truth_h = np.clip(obj['box'][3], 0, img_h)
            norm_x = int(truth_x*.99*cell_w/img_w)
            norm_y = int(truth_y*.99*cell_h/img_h)
            norm_w = int(truth_w/img_w)
            norm_h = int(truth_h/img_h)
            target[ind_img, norm_y, norm_x] = \
                np.concatenate(([1, norm_x%1, norm_y%1, norm_w, norm_h], one_hot_class_id))
    target = target.reshape(len(annotation_list), -1)
    return target, class_mapping
