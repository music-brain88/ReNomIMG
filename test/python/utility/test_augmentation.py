import os
import sys
import shutil
import pytest
import numpy as np
import inspect
from PIL import Image

from renom_img.api.utility.augmentation.process import contrast_norm
from renom_img.api.utility.augmentation.process import shift
from renom_img.api.utility.augmentation.process import rotate, flip, white_noise
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

    rescale(y, img.size, (1, 1))
    draw_box(x[0], y[0]).save(
        './outputs/test_augmentation_detection_{}1.png'.format(method.__name__))


# Test of augmentations for classification.
@pytest.mark.parametrize('method, kwargs', [
    [shift, {"horizontal": 50, "vertivcal": 50}],
    [rotate, {}],
    [flip, {}],
    [white_noise, {"std": 10}],
    [contrast_norm, {"alpha": [0.5, 1.0]}],
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
