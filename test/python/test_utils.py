import os
import sys
import shutil
import pytest

import numpy as np
from renom_img.api.utility.augmentation.process import contrast_norm
from PIL import Image


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
