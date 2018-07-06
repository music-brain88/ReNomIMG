import numpy as np
import inspect
import pytest
import types

from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.detection.yolo_v2 import Yolov2
from renom_img.api.model.vgg import VGG16
from renom_img.api.utility.augmentation import Augmentation


@pytest.mark.parametrize("algo", [
    Yolov1,
    Yolov2,
])
def test_detection_model_implementation(algo):
    # 1. Check if the model can be instantiate only giving nothing.
    try:
        model = algo()
    except Exception as e:
        # Model have to be initialized without no argument for using it with trained weight.
        raise Exception("Model can initialize without no argument.")

    methods = {k: v for k, v in inspect.getmembers(model) if inspect.ismethod(v)}

    # 2. Check function names and their arguments.
    method_list = {
        "__init__": [
            ["class_map", list],
            ["imsize", tuple],
            ["load_pretrained_weight", bool],
            ["train_whole_network", bool]
        ],
        "loss": ["x", "y"],
        "fit": ["train_img_path_list",
                "train_annotation_list",
                ["valid_img_path_list", type(None)],
                ["valid_annotation_list", type(None)],
                ["epoch", int],
                ["batch_size", int],
                ["augmentation", type(None)],
                ["callback_end_epoch", type(None)]
                ],
        "get_bbox": ["z"],
        "predict": [
            "img_list"
        ],
        "get_optimizer": [
            ["current_epoch", type(None)],
            ["total_epoch", type(None)],
            ["current_batch", type(None)],
            ["total_batch", type(None)],
        ],
        "preprocess": [
            "x"
        ],
        "build_data": [],
        "regularize": [],
    }

    for k, v in method_list.items():
        last_checked_index = -1
        assert k in methods
        args = inspect.getargspec(getattr(model, k))
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
                    raise ValueError("Argument '{}' is not implemented.".format(a))

            assert index > last_checked_index, \
                "The order of arguments are not correct."
            last_checked_index = index

    # 3. Check serializable attributes.
    serializables = [
        "class_map",
        "imsize",
        "num_class"
    ]
    for s in serializables:
        assert s in algo.SERIALIZED


@pytest.mark.parametrize("algo", [
    VGG16
])
def test_classification_model_implementation(algo):
    # 1. Check if the model can be instantiate only giving nothing.
    try:
        model = algo()
    except Exception as e:
        # Model have to be initializable without no argument for using it with trained weight.
        raise Exception("Model have to be  initializable without no argument.")

    methods = {k: v for k, v in inspect.getmembers(model) if inspect.ismethod(v)}

    # 2. Check function names and their arguments.
    method_list = {
        "__init__": [
            ["class_map", list],
            ["imsize", tuple],
            ["load_pretrained_weight", bool],
            ["train_whole_network", bool]
        ],
        "loss": ["x", "y"],
        "fit": ["train_img_path_list",
                "train_annotation_list",
                ["valid_img_path_list", type(None)],
                ["valid_annotation_list", type(None)],
                ["epoch", int],
                ["batch_size", int],
                ["augmentation", type(None)],
                ["callback_end_epoch", type(None)]
                ],
        "get_bbox": ["z"],
        "predict": [
            "img_list"
        ],
        "get_optimizer": [
            ["current_epoch", type(None)],
            ["total_epoch", type(None)],
            ["current_batch", type(None)],
            ["total_batch", type(None)],
        ],
        "preprocess": [
            "x"
        ],
        "build_data": [],
        "regularize": [],
    }

    for k, v in method_list.items():
        last_checked_index = -1
        assert k in methods
        args = inspect.getargspec(getattr(model, k))
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
                    raise ValueError("Argument '{}' is not implemented.".format(a))

            print(index, last_checked_index)
            assert index > last_checked_index, \
                "The order of arguments are not correct."
            last_checked_index = index

    # 3. Check serializable attributes.
    serializables = [
        "class_map",
        "imsize",
        "num_class"
    ]
    for s in serializables:
        assert s in algo.SERIALIZED
