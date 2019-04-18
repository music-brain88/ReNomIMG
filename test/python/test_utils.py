import os
import sys
import shutil
import pytest
import numpy as np
import inspect
from PIL import Image, ImageDraw
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.evaluate import EvaluatorClassification
from renom_img.api.utility.evaluate import EvaluatorDetection
from renom_img.api.utility.evaluate import EvaluatorSegmentation
from renom_img.api.utility.augmentation.process import contrast_norm
from renom_img.api.utility.augmentation.process import shift
from renom_img.api.utility.augmentation.process import *

from renom_img.api.utility.target import DataBuilderClassification, DataBuilderDetection, DataBuilderSegmentation

from renom_img.api.utility.misc.display import draw_box
from renom_img.api.utility.box import rescale


def create_points(size, pairs=2):
    p = []
    for n in range(pairs):
        i = np.minimum(1,n)
        a = np.random.randint(size[i]//2)
        b = np.random.randint(a,size[i])
        p.extend([a,b])
    return p


def create_seg_data(size, img_name, lbl_name):
    im = Image.new('RGB', size)

    p_x0,p_x1,p_y0,p_y1,p_z0,p_z1 = create_points(size, pairs=3)
    e_x0,e_x1,e_y0,e_y1 = create_points(size,pairs=2)
    r_x0,r_x1,r_y0,r_y1 = create_points(size,pairs=2)

    fill_p = tuple(np.random.randint(1,255,3))
    fill_e = tuple(np.random.randint(1,255,3))
    fill_r = tuple(np.random.randint(1,255,3))

    draw = ImageDraw.Draw(im)
    draw.polygon([(p_x0, p_y0), (p_x1, p_y1), (p_z0,p_z1)], fill=fill_p)
    draw.rectangle([(r_x0,r_y0),(r_x1,r_y1)], fill=fill_r)
    draw.ellipse([(e_x0,e_y0),(e_x1,e_y1)], fill=fill_e)

    im.save(img_name)

    lab = Image.new('L', size)

    draw = ImageDraw.Draw(lab)
    draw.polygon([(p_x0, p_y0), (p_x1, p_y1), (p_z0,p_z1)], fill=3)
    draw.rectangle([(r_x0,r_y0),(r_x1,r_y1)], fill=2)
    draw.ellipse([(e_x0,e_y0),(e_x1,e_y1)], fill=1)

    lab.save(lbl_name)


def delete_seg_data(img_name, lbl_name):
    if os.path.exists(img_name):
        os.remove(img_name)
    if os.path.exists(lbl_name):
        os.remove(lbl_name)


def load_img(img_path):
    img = Image.open(img_path)
    img.convert('RGB')
    x = np.array(img).transpose(2, 0, 1).astype(np.float)
    x = np.expand_dims(x, axis=0)
    return x, img


def load_seg_label(label_path):
    l = np.array(Image.open(label_path))
    n_class = len(np.unique(l))
    y = []
    annot = np.zeros((n_class, l.shape[0], l.shape[1]))
    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            if int(l[i][j]) >= n_class:
                annot[n_class - 1, i, j] = 1
            else:
                annot[int(l[i][j]), i, j] = 1
    y.append(annot)
    return y, l


def same_shape(x1, x2, transform):
    assert x1.shape == x2.shape, "Shapes do not match for {} transformation".format(
        transform.__name__)


def same_all(x1, x2, transform):
    assert x1 == x2, "Items are not equal for {} transformation".format(
        transform.__name__)


@pytest.fixture(scope='session', autouse=True)
def scope_session():
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
    os.mkdir('outputs')


# Test of augmentations for detection.
@pytest.mark.parametrize('image_path', [
    './renom.png',
    './renom.png',
    './renom.png',
    './renom.png',
    './renom.png'
])
@pytest.mark.parametrize('method, kwargs', [
    [shift, {"horizontal": 50, "vertivcal": 50}],
    [rotate, {}],
    [flip, {}],
    [white_noise, {"std": 10}],
    [contrast_norm, {"alpha": [0.5, 1.0]}],
    [random_crop, {}],
    [random_brightness, {}],
    [random_hue, {}],
    [random_saturation, {}],
    [random_lighting, {}],
    [random_expand, {}],
    [color_jitter, {}],
    [color_jitter, {"h":(0.85,1.05),"s":(0.85,1.05),"v":(0.95,1.05)}],
    [shear, {}],
    [horizontalflip, {}],
    [verticalflip, {}],
    [center_crop, {}],
    [distortion, {}],
])
def test_augmentation_process_detection(method, kwargs, image_path):
    img = Image.open(image_path)
    img.convert('RGB')
    x = np.array(img).transpose(2, 0, 1).astype(np.float)
    x = np.expand_dims(x, axis=0)
    y = [[
        {"box": [380, 180, 40, 50], "class":0, "name": "test1"},
        {"box": [60, 80, 130, 70], "class":1, "name": "test2"}
    ]]
    rescale(y, img.size, (1, 1))
    draw_box(x[0], y[0]).save(
        './outputs/test_augmentation_detection_{}0.png'.format(method.__name__))

    rescale(y, (1, 1), img.size)
    x, y = method(x, y, mode="detection", **kwargs)

    rescale(y, (x[0].shape[2], x[0].shape[1]), (1, 1))
    draw_box(x[0], y[0]).save(
        './outputs/test_augmentation_detection_{}1.png'.format(method.__name__))


# Test of augmentations for classification.
@pytest.mark.parametrize('image_path', [
    './renom.png',
    './renom.png',
    './renom.png',
    './renom.png',
    './renom.png'
])
@pytest.mark.parametrize('method, kwargs', [
    [shift, {"horizontal": 50, "vertivcal": 50}],
    [rotate, {}],
    [flip, {}],
    [white_noise, {"std": 10}],
    [contrast_norm, {"alpha": [0.5, 1.0]}],
    [random_crop, {}],
    [random_hue, {}],
    [random_brightness, {}],
    [random_saturation, {}],
    [random_lighting, {}],
    [random_expand, {}],
    [color_jitter, {}],
    [color_jitter, {"h":(0.85,1.05),"s":(0.85,1.05),"v":(0.95,1.05)}],
    [shear, {}],
    [horizontalflip, {}],
    [verticalflip, {}],
    [center_crop, {}],
    [distortion, {}],
])
def test_augmentation_process_classification(method, kwargs, image_path):
    img = Image.open(image_path)
    img.convert('RGB')
    x = np.array(img).transpose(2, 0, 1).astype(np.float)
    x = np.expand_dims(x, axis=0)
    y = [[0]]
    Image.fromarray(x[0].transpose(1, 2, 0).astype(np.uint8)).save(
        './outputs/test_augmentation_classification_{}0.png'.format(method.__name__))
    x, y = method(x, y, mode="classification", **kwargs)
    Image.fromarray(x[0].transpose(1, 2, 0).astype(np.uint8)).save(
        './outputs/test_augmentation_classification_{}1.png'.format(method.__name__))


# Test segmentation augmentation methods that should not affect label at all
@pytest.mark.parametrize('size,image,label', [
    [(240,240), 'img_1.jpg', 'lbl_1.png' ],
    [(240,300), 'img_2.jpg', 'lbl_2.png' ],
    [(300,240), 'img_3.jpg', 'lbl_3.png' ],
    [(240,500), 'img_4.jpg', 'lbl_4.png' ],
    [(500,240), 'img_5.jpg', 'lbl_5.png' ]
])
@pytest.mark.parametrize('method, kwargs', [
    [white_noise, {"std": 1.5}],
    [contrast_norm, {"alpha": [0.5, 1.0]}],
    [random_hue, {}],
    [random_brightness, {}],
    [random_saturation, {}],
    [random_lighting, {}],
    [color_jitter, {}],
    [color_jitter, {"h":(0.85,1.05),"s":(0.85,1.05),"v":(0.95,1.05)}]
])
def test_augmentation_process_segmentation_noise(method, kwargs, size, image, label):
    create_seg_data(size, image, label)
    x, img = load_img(image)
    y, l = load_seg_label(label)

    x_aug, y_aug = method(x, y, mode="segmentation", **kwargs)

    same_shape(x_aug[0], x[0], method)
    same_shape(y_aug[0], y[0], method)
    same_all(y_aug, y, method)
    delete_seg_data(image, label)


# Test segmentation augmentation shift method
@pytest.mark.parametrize('size,image,label', [
    [(240,240), 'img_1.jpg', 'lbl_1.png' ],
    [(240,300), 'img_2.jpg', 'lbl_2.png' ],
    [(300,240), 'img_3.jpg', 'lbl_3.png' ],
    [(240,500), 'img_4.jpg', 'lbl_4.png' ],
    [(500,240), 'img_5.jpg', 'lbl_5.png' ]
])
@pytest.mark.parametrize('method, kwargs', [
    [shift, {"horizontal": 10, "vertivcal": 10}],
    [random_crop, {}]
])
def test_augmentation_process_segmentation_shift(method, kwargs, size, image, label):
    create_seg_data(size, image, label)
    x, img = load_img(image)
    y, l = load_seg_label(label)

    x_aug, y_aug = method(x, y, mode="segmentation", **kwargs)

    same_shape(x_aug[0], x[0], method)
    same_shape(y_aug[0], y[0], method)
    assert np.count_nonzero(y_aug[0]) <= np.count_nonzero(y[0])
    assert (y_aug[0] == 0).sum() >= (y[0] == 0).sum()
    delete_seg_data(image, label)


@pytest.mark.parametrize('size,image,label', [
    [(240,240), 'img_1.jpg', 'lbl_1.png' ],
    [(240,300), 'img_2.jpg', 'lbl_2.png' ],
    [(300,240), 'img_3.jpg', 'lbl_3.png' ],
    [(240,500), 'img_4.jpg', 'lbl_4.png' ],
    [(500,240), 'img_5.jpg', 'lbl_5.png' ]
])
@pytest.mark.parametrize('method, kwargs', [
    [center_crop, {}]
])
def test_augmentation_process_segmentation_centercrop(method, kwargs, size, image, label):
    create_seg_data(size, image, label)
    x, img = load_img(image)
    y, l = load_seg_label(label)

    x_aug, y_aug = method(x, y, mode="segmentation", **kwargs)

    assert x_aug[0].shape[1:] == y_aug[0].shape[1:], "Shapes do not match"
    assert y_aug[0].shape[1:] == (224, 224)
    for i in np.unique(y_aug[0]):
        assert (y_aug[0] == i).sum() <= (y[0] == i).sum()
    delete_seg_data(image, label)


# Test segmentation augmentation shift method
@pytest.mark.parametrize('size,image,label', [
    [(240,240), 'img_1.jpg', 'lbl_1.png' ],
    [(240,300), 'img_2.jpg', 'lbl_2.png' ],
    [(300,240), 'img_3.jpg', 'lbl_3.png' ],
    [(240,500), 'img_4.jpg', 'lbl_4.png' ],
    [(500,240), 'img_5.jpg', 'lbl_5.png' ]
])
@pytest.mark.parametrize('method, kwargs', [
    [random_expand, {}]
])
def test_augmentation_process_segmentation_randomexpand(method, kwargs, size, image, label):
    create_seg_data(size, image, label)
    x, img = load_img(image)
    y, l = load_seg_label(label)

    x_aug, y_aug = method(x, y, mode="segmentation", **kwargs)

    assert x_aug[0].shape[1:] == y_aug[0].shape[1:], "Shapes do not match"
    assert np.count_nonzero(y_aug[0]) <= np.count_nonzero(y[0])
    assert (y_aug[0] == 0).sum() >= (y[0] == 0).sum()
    delete_seg_data(image, label)


@pytest.mark.parametrize('size,image,label', [
    [(240,240), 'img_1.jpg', 'lbl_1.png' ],
    [(240,300), 'img_2.jpg', 'lbl_2.png' ],
    [(300,240), 'img_3.jpg', 'lbl_3.png' ],
    [(240,500), 'img_4.jpg', 'lbl_4.png' ],
    [(500,240), 'img_5.jpg', 'lbl_5.png' ]
])
@pytest.mark.parametrize('method, kwargs', [
    [shear, {}]
])
def test_augmentation_process_segmentation_shear(method, kwargs, size, image, label):
    create_seg_data(size, image, label)
    x, img = load_img(image)
    y, l = load_seg_label(label)

    x_aug, y_aug = method(x, y, mode="segmentation", **kwargs)

    assert x_aug[0].shape[1:] == y_aug[0].shape[1:], "Shapes do not match"
    delete_seg_data(image, label)


@pytest.mark.parametrize('size,image,label', [
    [(240,240), 'img_1.jpg', 'lbl_1.png' ],
    [(240,300), 'img_2.jpg', 'lbl_2.png' ],
    [(300,240), 'img_3.jpg', 'lbl_3.png' ],
    [(240,500), 'img_4.jpg', 'lbl_4.png' ],
    [(500,240), 'img_5.jpg', 'lbl_5.png' ]
])
@pytest.mark.parametrize('method, kwargs', [
    [rotate, {}],
    [distortion, {}]
])
def test_augmentation_process_segmentation_rotate(method, kwargs, size, image, label):
    create_seg_data(size, image, label)
    x, img = load_img(image)
    y, l = load_seg_label(label)

    x_aug, y_aug = method(x, y, mode="segmentation", **kwargs)

    same_shape(x_aug[0], x[0], method)
    same_shape(y_aug[0], y[0], method)
    for i in range(len(np.unique(y[0], return_counts=True))):
        assert np.unique(y[0], return_counts=True)[i].all() == \
            np.unique(y_aug[0], return_counts=True)[i].all(), \
            "{} {} {}".format(i, np.unique(y[0], return_counts=True)[
                              i], np.unique(y_aug[0], return_counts=True)[i])
    delete_seg_data(image, label)


# Test segmentation augmentation methods that flip image and label
@pytest.mark.parametrize('size,image,label', [
    [(240,240), 'img_1.jpg', 'lbl_1.png' ],
    [(240,300), 'img_2.jpg', 'lbl_2.png' ],
    [(300,240), 'img_3.jpg', 'lbl_3.png' ],
    [(240,500), 'img_4.jpg', 'lbl_4.png' ],
    [(500,240), 'img_5.jpg', 'lbl_5.png' ]
])
@pytest.mark.parametrize('method, kwargs', [
    [flip, {}],
    [horizontalflip, {}],
    [verticalflip, {}]
])
def test_augmentation_process_segmentation_flip(method, kwargs, size, image, label):
    create_seg_data(size, image, label)
    x, img = load_img(image)
    y, l = load_seg_label(label)

    x_aug, y_aug = method(x, y, mode="segmentation", **kwargs)

    same_shape(x_aug[0], x[0], method)
    same_shape(y_aug[0], y[0], method)

    if method.__name__=='flip':
        assert x_aug[0].all() == x[0][:, ::-1, ::-1].all()
        assert y_aug[0].all() == y[0][:, ::-1, ::-1].all()
    elif method.__name__=='horizontalflip':
        assert x_aug[0].all() == x[0][:, :, ::-1].all()
        assert y_aug[0].all() == y[0][:, :, ::-1].all()
    elif method.__name__=='verticalflip':
        assert x_aug[0].all() == x[0][:, ::-1, :].all()
        assert y_aug[0].all() == y[0][:, ::-1, :].all()
    delete_seg_data(image, label)


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


@pytest.mark.parametrize('path, error', [
    ['voc.xml', False],
    ['illiegal_voc.xml', True],
])
def test_parse_xml_detection(path, error):
    try:
        ret = parse_xml_detection([path], num_thread=1)
        assert not error
    except Exception as e:
        print(e)
        assert error
