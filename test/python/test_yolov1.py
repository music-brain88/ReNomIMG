import os
import numpy as np
import inspect
import pytest
import types

from renom_img.api.detection.yolo_v1 import Yolov1


def test_has_pretrained_weight():
    weight_path = "%s.h5" % Yolov1.__name__
    if os.path.exists(weight_path):
        os.remove(weight_path)
    N = 10
    IMG = (3, 64, 64)
    x = np.random.rand(N, *IMG)
    model = Yolov1(load_pretrained_weight=True)
    os.remove(weight_path)
