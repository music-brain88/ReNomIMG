import os
import sys
import shutil
import pytest

import numpy as np
from renom_img.api.utility.augmentation.process import contrast_norm
from PIL import Image
from renom_img.api.utility.evaluate import EvaluatorDetection


@pytest.fixture(scope='session', autouse=True)
def scope_session():
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
    os.mkdir('outputs')


def test_contrast_norm():
    img = Image.open('./renom.png')
    img.convert('RGB')
    img = np.array(img).transpose(2, 0, 1).astype(np.float)
    x = np.array([img])
    y = [1]
    alpha = 0.5
    answer = alpha * (x - 128) + 128
    result = contrast_norm(x, y, alpha=alpha)
    Image.fromarray(np.uint8(result[0][0].transpose(1, 2, 0))
                    ).save('./outputs/test_contrast_norm.png')
    assert (answer == result[0]).any()
    assert y == result[1]


@pytest.mark.parametrize('pred, gt', [
    [[[{'box': [10, 20, 60, 80], 'score': 0.8, 'class': 0},
        {'box': [70, 90, 120, 110], 'score': 0.9, 'class': 1},
        {'box': [10, 20, 30, 40], 'score': 0.9, 'class': 1}],
      [{'box': [20, 40, 70, 90], 'score': 0.8, 'class': 0}]],
     [[{'box': [15, 25, 55, 70], 'class': 0},
       {'box': [80, 95, 125, 105], 'class': 1}],
      [{'box': [35, 45, 75, 95], 'class': 0}]]]
])
def test_evaluator_mAP(pred, gt):
    evalDetection = EvaluatorDetection(pred, gt)
    mAP = evalDetection.mAP()
    assert mAP == 0.75


@pytest.mark.parametrize('pred, gt', [
    [[[{'box': [40, 30, 60, 40], 'score': 0.8, 'class': 0},
        {'box': [70, 90, 40, 20], 'score': 0.9, 'class': 1},
        {'box': [10, 20, 30, 40], 'score': 0.9, 'class': 1}],
      [{'box': [80, 100, 60, 40], 'score': 0.8, 'class': 0}]],
     [[{'box': [45, 30, 60, 50], 'class': 0},
       {'box': [70, 95, 40, 30], 'class': 1}],
      [{'box': [80, 90, 60, 50], 'class': 0}]]]
])
def test_evaluator_AP(pred, gt):
    evalDetection = EvaluatorDetection(pred, gt)
    AP = evalDetection.AP()
    assert round(AP[0], 3) == 1
    assert AP[1] == 0.5


@pytest.mark.parametrize('pred, gt', [
    [[[{'box': [40, 30, 60, 40], 'score': 0.8, 'class': 0},
        {'box': [70, 90, 40, 20], 'score': 0.9, 'class': 1},
        {'box': [10, 20, 30, 40], 'score': 0.9, 'class': 1}],
      [{'box': [80, 100, 60, 40], 'score': 0.8, 'class': 0}]],
     [[{'box': [45, 30, 60, 50], 'class': 0},
       {'box': [70, 95, 40, 30], 'class': 1}],
      [{'box': [80, 90, 60, 50], 'class': 0}]]]
])
def test_evaluator_mean_iou(pred, gt):
    evalDetection = EvaluatorDetection(pred, gt)
    mean_iou = evalDetection.mean_iou()
    assert mean_iou == 0.665


@pytest.mark.parametrize('pred, gt', [
    [[[{'box': [40, 30, 60, 40], 'score': 0.8, 'class': 0},
        {'box': [70, 90, 40, 20], 'score': 0.9, 'class': 1},
        {'box': [10, 20, 30, 40], 'score': 0.9, 'class': 1}],
      [{'box': [80, 100, 60, 40], 'score': 0.8, 'class': 0}]],
     [[{'box': [45, 30, 60, 50], 'class': 0},
       {'box': [70, 95, 40, 30], 'class': 1}],
      [{'box': [80, 90, 60, 50], 'class': 0}]]]
])
def test_evaluator_iou(pred, gt):
    evalDetection = EvaluatorDetection(pred, gt)
    iou = evalDetection.iou()
    assert iou[0] == 0.662
    assert iou[1] == 0.667
