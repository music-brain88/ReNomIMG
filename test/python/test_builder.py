import os
import sys
import shutil
import pytest
import numpy as np
import inspect
from PIL import Image
from renom_img.api.utility.evaluate import EvaluatorClassification
from renom_img.api.utility.evaluate import EvaluatorDetection
from renom_img.api.utility.evaluate import EvaluatorSegmentation
from renom_img.api.utility.augmentation.process import contrast_norm
from renom_img.api.utility.augmentation.process import shift
from renom_img.api.utility.augmentation.process import rotate, flip, white_noise
from renom_img.api.utility.target import DataBuilderClassification, DataBuilderDetection, DataBuilderSegmentation

from renom_img.api.utility.misc.display import draw_box
from renom_img.api.utility.box import rescale


@pytest.fixture(scope='session', autouse=True)
def scope_session():
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
    os.mkdir('outputs')


@pytest.mark.parametrize('method', [
    DataBuilderClassification,
    DataBuilderDetection,
    DataBuilderSegmentation
])
def test_target_bulder_implementation(method):
    class_map = ["test1", "test2"]
    imsize = (224, 224)

    # 1. Check instantiation.
    target_builder = method(class_map, imsize)

    # 2. Check functions name and their arguments.
    methods = {k: v for k, v in inspect.getmembers(target_builder) if inspect.ismethod(v)}

    method_list = {
        "build": [
            "img_path_list",
            "annotation_list",
            ["augmentation", type(None)],
        ],
    }
    for k, v in method_list.items():
        last_checked_index = -1
        assert k in methods
        args = inspect.getargspec(getattr(target_builder, k))
        for i, a in enumerate(v):
            if isinstance(a, list):
                try:
                    index = args.args.index(a[0])
                except ValueError as e:
                    raise ValueError("Argument '{}' is not implemented.".format(a[0]))
                assert a[1] == type(args.defaults[index - (len(args.args) - len(args.defaults))]), \
                    "Default argument type miss matched."
            else:
                try:
                    index = args.args.index(a)
                except ValueError as e:
                    raise ValueError(
                        "Argument '{}' is not implemented. There is {}".format(a, args.args))

            assert index > last_checked_index, \
                "The order of arguments are not correct."
            last_checked_index = index
