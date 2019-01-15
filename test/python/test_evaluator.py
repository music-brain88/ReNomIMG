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
