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
from renom_img.api.utility.augmentation.process import *
from renom_img.api.utility.target import DataBuilderClassification, DataBuilderDetection, DataBuilderSegmentation

from renom_img.api.utility.misc.display import draw_box
from renom_img.api.utility.box import rescale


@pytest.fixture(scope='session', autouse=True)
def scope_session():
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
    os.mkdir('outputs')


# Test of augmentations for detection.
@pytest.mark.parametrize('method, kwargs', [
    [shift, {"horizontal": 50, "vertivcal": 50}],
    [rotate, {}],
    [flip, {}],
    [white_noise, {"std": 10}],
    [contrast_norm, {"alpha": [0.5, 1.0]}],
    [random_crop, {}],
    [random_brightness,{}],
    [random_hue,{}],
    [random_saturation,{}],
    [random_lighting,{}],
    [random_expand,{}],
])
def test_augmentation_process_detection(method, kwargs):
    img = Image.open('./renom.png')
    img.convert('RGB')
    x = np.array(img).transpose(2, 0, 1).astype(np.float)
    x = np.expand_dims(x, axis=0)
    y = [[
        {"box": [100, 60, 40, 50], "class":0, "name": "test1"},
        {"box": [40, 60, 100, 50], "class":1, "name": "test2"}
    ]]
    rescale(y, img.size, (1, 1))
    draw_box(x[0], y[0]).save(
        './outputs/test_augmentation_detection_{}0.png'.format(method.__name__))

    rescale(y, (1, 1), img.size)
    x, y = method(x, y, mode="detection", **kwargs)

    rescale(y, (x[0].shape[2],x[0].shape[1]), (1, 1))
    draw_box(x[0], y[0]).save(
        './outputs/test_augmentation_detection_{}1.png'.format(method.__name__))


# Test of augmentations for classification.
@pytest.mark.parametrize('method, kwargs', [
    [shift, {"horizontal": 50, "vertivcal": 50}],
    [rotate, {}],
    [flip, {}],
    [white_noise, {"std": 10}],
    [contrast_norm, {"alpha": [0.5, 1.0]}],
    [random_crop,{}],
    [random_hue,{}],
    [random_brightness,{}],
    [random_saturation,{}],
    [random_lighting,{}],
    [random_expand,{}],
])
def test_augmentation_process_classification(method, kwargs):
    img = Image.open('./renom.png')
    img.convert('RGB')
    x = np.array(img).transpose(2, 0, 1).astype(np.float)
    x = np.expand_dims(x, axis=0)
    y = [[0]]
    Image.fromarray(x[0].transpose(1, 2, 0).astype(np.uint8)).save(
        './outputs/test_augmentation_classification_{}0.png'.format(method.__name__))
    x, y = method(x, y, mode="classification", **kwargs)
    Image.fromarray(x[0].transpose(1, 2, 0).astype(np.uint8)).save(
        './outputs/test_augmentation_classification_{}1.png'.format(method.__name__))


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


@pytest.mark.parametrize('pred, gt', [
    [[[{'box': [40, 30, 60, 40], 'score': 0.8, 'class': 0, 'name': 'dog'},
        {'box': [70, 90, 40, 20], 'score': 0.9, 'class': 1, 'name': 'cat'},
        {'box': [10, 20, 30, 40], 'score': 0.9, 'class': 1, 'name': 'cat'}],
        [{'box': [80, 100, 60, 40], 'score': 0.8, 'class': 0, 'name': 'dog'}]],
        [[{'box': [45, 30, 60, 50], 'class': 0, 'name': 'dog'},
            {'box': [70, 95, 40, 30], 'class': 1, 'name': 'cat'}],
            [{'box': [80, 90, 60, 50], 'class': 0, 'name': 'dog'}]]]
])
def test_evaluator_mAP(pred, gt):
    evalDetection = EvaluatorDetection(pred, gt)
    mAP = evalDetection.mAP()
    assert mAP == 0.75


@pytest.mark.parametrize('pred, gt', [
    [[[{'box': [40, 30, 60, 40], 'score': 0.8, 'class': 0, 'name': 'dog'},
        {'box': [70, 90, 40, 20], 'score': 0.9, 'class': 1, 'name': 'cat'},
        {'box': [10, 20, 30, 40], 'score': 0.9, 'class': 1, 'name': 'cat'}],
        [{'box': [80, 100, 60, 40], 'score': 0.8, 'class': 0, 'name': 'dog'}]],
        [[{'box': [45, 30, 60, 50], 'class': 0, 'name': 'dog'},
            {'box': [70, 95, 40, 30], 'class': 1, 'name': 'cat'}],
            [{'box': [80, 90, 60, 50], 'class': 0, 'name': 'dog'}]]]
])
def test_evaluator_AP(pred, gt):
    evalDetection = EvaluatorDetection(pred, gt)
    AP = evalDetection.AP()
    assert round(AP['dog'], 3) == 1
    assert AP['cat'] == 0.5


@pytest.mark.parametrize('pred, gt', [
    [[[{'box': [40, 30, 60, 40], 'score': 0.8, 'class': 0, 'name': 'dog'},
        {'box': [70, 90, 40, 20], 'score': 0.9, 'class': 1, 'name': 'cat'},
        {'box': [10, 20, 30, 40], 'score': 0.9, 'class': 1, 'name': 'cat'}],
        [{'box': [80, 100, 60, 40], 'score': 0.8, 'class': 0, 'name': 'dog'}]],
        [[{'box': [45, 30, 60, 50], 'class': 0, 'name': 'dog'},
            {'box': [70, 95, 40, 30], 'class': 1, 'name': 'cat'}],
            [{'box': [80, 90, 60, 50], 'class': 0, 'name': 'dog'}]]]
])
def test_evaluator_mean_iou(pred, gt):
    evalDetection = EvaluatorDetection(pred, gt)
    mean_iou = evalDetection.mean_iou()
    assert mean_iou == 0.665


@pytest.mark.parametrize('pred, gt', [
    [[[{'box': [40, 30, 60, 40], 'score': 0.8, 'class': 0, 'name': 'dog'},
        {'box': [70, 90, 40, 20], 'score': 0.9, 'class': 1, 'name': 'cat'},
        {'box': [20, 20, 30, 40], 'score': 0.9, 'class': 1, 'name': 'cat'}],
        [{'box': [80, 100, 60, 40], 'score': 0.8, 'class': 0, 'name': 'dog'}]],
        [[{'box': [45, 30, 60, 50], 'class': 0, 'name': 'dog'},
            {'box': [70, 95, 40, 30], 'class': 1, 'name': 'cat'}],
            [{'box': [80, 90, 60, 50], 'class': 0, 'name': 'dog'}]]]
])
def test_evaluator_iou(pred, gt):
    evalDetection = EvaluatorDetection(pred, gt)
    iou = evalDetection.iou()
    assert iou['dog'] == 0.662
    assert iou['cat'] == 0.667
