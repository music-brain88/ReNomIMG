import numpy as np
import inspect
import pytest
import types

from renom.cuda import set_cuda_active, release_mem_pool

from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.detection.yolo_v2 import Yolov2, AnchorYolov2
from renom_img.api.detection.ssd import SSD

from renom_img.api.classification.vgg import VGG16, VGG19
from renom_img.api.classification.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from renom_img.api.classification.resnext import ResNeXt50, ResNeXt101
from renom_img.api.classification.inception import InceptionV1, InceptionV2, InceptionV3, InceptionV4
from renom_img.api.classification.densenet import DenseNet121, DenseNet169, DenseNet201
from renom_img.api.segmentation.unet import UNet
from renom_img.api.segmentation.fcn import FCN16s, FCN8s, FCN32s
from renom_img.api.segmentation.ternausnet import TernausNet

from renom_img.api.utility.augmentation import Augmentation
from renom_img.api.utility.augmentation.process import HorizontalFlip, ContrastNorm, WhiteNoise
from renom_img.api.utility.load import parse_xml_detection

set_cuda_active(True)


@pytest.mark.parametrize("algo", [
    Yolov1,
    Yolov2,
    SSD
])
def test_detection_model_implementation(algo):
    # 1. Check if the model can be instantiate only giving nothing.
    try:
        model = algo()
    except Exception as e:
        # Model have to be initialized without argument for using it with trained weight.
        raise Exception("Model can not be initialized without arguments.")

    methods = {k: v for k, v in inspect.getmembers(model) if inspect.ismethod(v)}

    # 2. Check function names and their arguments.
    method_list = {
        "__init__": [
            ["class_map", type(None)],
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
            "img_list",
            ["batch_size", type(1)],
            "score_threshold",
            "nms_threshold"
        ],
        "preprocess": [
            "x"
        ],
        "build_data": [],
        "regularize": []
    }

    for k, v in method_list.items():
        last_checked_index = -1
        assert k in methods
        args = inspect.getfullargspec(getattr(model, k))
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
                "The order of arguments are not correct in {}.".format(k)
            last_checked_index = index

    # 3. Check serializable attributes.
    serializables = [
        "class_map",
        "imsize",
        "num_class"
    ]
    for s in serializables:
        assert s in algo.SERIALIZED

    # 4. Check fit function.
    test_imgs = [
        "utility/voc.jpg",
        "utility/voc.jpg",
    ]
    test_xmls = [
        "utility/voc.xml",
        "utility/voc.xml"
    ]
    test_annotation, class_map = parse_xml_detection(test_xmls)
    if algo is Yolov2:
        # Yolo needs anchor.
        model = algo(class_map, anchor=AnchorYolov2(np.array([[0.2, 0.3],[0.3,0.4]]),(224, 224)))
    else:
        model = algo(class_map)
    model.fit(test_imgs, test_annotation, test_imgs, test_annotation, batch_size=2, epoch=2)

    # Predict
    model.predict(test_imgs)


@pytest.mark.parametrize("algo", [
    VGG16,
    VGG19,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNeXt50,
    ResNeXt101
    # InceptionV1,
    # InceptionV2,
    # InceptionV3,
    # InceptionV4,
    # DenseNet121,
    # DenseNet169,
    # DenseNet201,
])
def test_classification_model_implementation(algo):
    release_mem_pool()
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
            ["class_map", type(None)],
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
        "predict": [
            "img_list"
        ],
        "preprocess": [
            "x"
        ],
        "regularize": []
    }

    for k, v in method_list.items():
        last_checked_index = -1
        assert k in methods
        args = inspect.getfullargspec(getattr(model, k))
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

    # 4. Check fit function.
    test_imgs = [
        "utility/voc.jpg",
        "utility/voc.jpg",
    ]
    test_annotation = [
        0, 0
    ]

    class_map = ["car"]
    model = algo(class_map)
    model.fit(test_imgs, test_annotation, test_imgs, test_annotation, batch_size=2, epoch=2)

    # Predict
    model.predict(test_imgs)


@pytest.mark.parametrize("algo", [
    UNet,
    FCN8s,
    FCN16s,
    FCN32s,
    TernausNet
])
def test_segmentation_model_implementation(algo):
    release_mem_pool()
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
            ["class_map", type(None)],
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
        "predict": [
            "img_list"
        ],
        "preprocess": [
            "x"
        ],
        "regularize": []
    }

    for k, v in method_list.items():
        last_checked_index = -1
        assert k in methods
        args = inspect.getfullargspec(getattr(model, k))
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

    # 4. Check fit function.
    test_imgs = [
        "utility/voc.jpg",
        "utility/voc.jpg",
    ]
    test_annotation = [
        "utility/segmentation_target.png",
        "utility/segmentation_target.png",
    ]

    class_map = ["background", "airplane"]
    aug = Augmentation([HorizontalFlip(), ContrastNorm(), WhiteNoise()])
    model = algo(class_map)
    class_weight = model.build_class_weight(test_annotation)
    model.fit(test_imgs, test_annotation, test_imgs, test_annotation,
              batch_size=2, epoch=2, augmentation=aug, class_weight=class_weight)

    # Predict
    model.predict(test_imgs)
