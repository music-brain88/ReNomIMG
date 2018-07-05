import os
import numpy as np
from renom_img.api.model.vgg import VGG16, VGG19

"""
Testing the weight will be downloaded and usable.
"""


def test_vgg16():
    if os.path.exists(VGG16.WEIGHT_PATH):
        os.remove(VGG16.WEIGHT_PATH)
    model = VGG16(load_weight=True)
    model = VGG16(num_class=10, load_weight=False)
    assert not model._layers[-1].params
    z = model(np.random.rand(1, 3, 224, 224))
    assert z.shape == (1, 10)


def test_vgg19():
    if os.path.exists(VGG19.WEIGHT_PATH):
        os.remove(VGG19.WEIGHT_PATH)
    model = VGG19(load_weight=True)
    model = VGG19(num_class=10, load_weight=False)
    assert not model._layers[-1].params
    z = model(np.random.rand(1, 3, 224, 224))
    assert z.shape == (1, 10)
