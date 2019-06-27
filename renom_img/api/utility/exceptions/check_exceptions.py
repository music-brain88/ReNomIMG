import numpy as np
import renom as rm
from renom_img.api.utility.exceptions.standards import *
from renom_img.api.utility.exceptions.exceptions import *


def check_for_common_init_params(class_map, imsize, load_pretrained_weight, train_whole_network, target_model):

    standard_obj = StandardInit()
    # value check
    standards = standard_obj.get_standards()
    try:
        if class_map is not None:
            assert all(isinstance(k, standards['class_map']['value'][0]) for k in class_map), "Invalid element in class_map {}. Please provide only string elements.".format(
                class_map, standards['class_map']['value'][0])

        if isinstance(imsize, tuple):
            assert all(isinstance(k, standards['imsize']['value'][0])
                       for k in imsize), "Invalid imsize values ({}). Please provide integer values.".format(imsize)
        else:
            assert isinstance(imsize, standards['imsize']['value'][0]
                              ), "Invalid imsize value ({}). Please provide an integer value.".format(imsize)

    except Exception as e:
        raise InvalidInputValueError(str(e))
    # type check
    try:
        assert type(class_map) in standards['class_map']['type'], "Invalid type for class_map argument. Please provide a {} type.".format(
            standards['class_map']['type'])
        assert type(imsize) in standards['imsize']['type'], "Invalid type for imsize argument. Please provide a {} type.".format(
            standards['imsize']['type'])
        assert type(load_pretrained_weight) in standards['load_pretrained_weight']['type'], "Invalid type for load_pretrained_weight argument. Please provide a {} type.".format(
            standards['load_pretrained_weight']['type'])
        assert type(train_whole_network) in standards['train_whole_network']['type'], "Invalid type for train_whole_network argument. Please provide a {} type.".format(
            standards['train_whole_network']['type'])
        assert isinstance(target_model, standards['target_model']['type']), "Invalid type for target_model argument. Please provide a {} type".format(
            standards['train_whole_network']['type'])
    except Exception as e:
        raise InvalidInputTypeError(str(e))

    # check range limit
    try:
        if class_map is not None:
            assert len(class_map) >= standards['class_map']['range'][0] and len(class_map) <= standards['class_map']['range'][1], "Invalid class_map length. Please provide a class_map with length between {} and {} elements".format(
                standards['class_map']['range'][0], standards['class_map']['range'][1])
        if type(imsize) is tuple:
            assert imsize[0] >= standards['imsize']['range'][0] and imsize[0] <= standards['imsize']['range'][1] and imsize[1] >= standards['imsize']['range'][0] and imsize[1] <= standards[
                'imsize']['range'][1], "Invalid imsize values. Please provide values between {} and {}".format(standards['imsize']['range'][0], standards['imsize']['range'][1])
        else:
            assert imsize >= standards['imsize']['range'][0] and imsize <= standards['imsize']['range'][1], "Invalid imsize value. Please provide a value between {} and {}".format(standards['imsize']['range'][0], standards['imsize']['range'][1])
    except Exception as e:
        raise InvalidInputValueError(str(e))


def check_resnet_init(plateau):
    standard_obj = StandardResNetInit()
    std = standard_obj.get_standards()

    try:
        assert type(plateau) in std['plateau']['type'], "Invalid plateau type. Please provide a {} type for the plateau argument.".format(std['plateau']['type'])
    except Exception as e:
        raise InvalidInputTypeError(str(e))


def check_resnext_init(plateau, cardinality):
    standard_obj = StandardResNeXtInit()
    std = standard_obj.get_standards()

    try:
        assert type(plateau) in std['plateau']['type'], "Invalid plateau type. Please provide a {} type for the plateau argument.".format(std['plateau']['type'])
        assert type(cardinality) in std['cardinality']['type'], "Invalid cardinality type. Please provide a {} type for the cardinality argument.".format(std['cardinality']['type'])
    except Exception as e:
        raise InvalidInputTypeError(str(e))
    try:
        assert cardinality >= std['cardinality']['range'][0] and cardinality <= std['cardinality']['range'][1], "Invalid cardinality value. Please provide a cardinality value between {} and {}.".format(std['cardinality']['range'][0], std['cardinality']['range'][1])
    except Exception as e:
        raise InvalidInputValueError(str(e))


def check_yolov1_init(cells, bbox):
    obj = StandardYolov1Init()
    std = obj.get_standards()

    try:
        assert type(cells) in std['cells']['type'], "Invalid type for cells argument. Please provide a {} type.".format(std['cells']['type'])
        assert type(bbox) in std['bbox']['type'], "Invalid type for bbox argument. Please provide a {} type.".format(std['bbox']['type'])
    except Exception as e:
        raise InvalidInputTypeError(str(e))

    try:
        if type(cells) is tuple:
            assert cells[0] >= std['cells']['range'][0] and cells[0] <= std['cells']['range'][1] and cells[
                1] >= std['cells']['range'][0] and cells[1] <= std['cells']['range'][1], "Invalid cells values. Please provide values between {} and {} for cells argument.".format(std['cells']['range'][0], std['cells']['range'][1])
        else:
            assert cells >= std['cells']['range'][0] and cells <= std['cells']['range'][1], "Invalid cells value. Please provide a value between {} and {} for cells argument.".format(std['cells']['range'][0], std['cells']['range'][1])

        assert bbox >= std['bbox']['range'][0] and bbox <= std['bbox']['range'][1], "Invalid bbox value. Please provide a value between {} and {} for the bbox argument.".format(std['bbox']['range'][0], std['bbox']['range'][1])
    except Exception as e:
        raise InvalidInputValueError(str(e))


def check_yolov2_init(imsize):
    obj = StandardYolov2Init()
    std = obj.get_standards()
    try:
        if type(imsize) is tuple:
            assert all(k % std['multiple'] == 0 for k in imsize), "Invalid imsize value. imsize for Yolov2 must be integer multiple of {}.".format(
                std['multiple'])
        else:
            assert imsize % std['multiple'] == 0, "Invalid imsize value. imsize for Yolov2 must be integer multiple of {}.".format(
                std['multiple'])
    except Exception as e:
        raise InvalidInputValueError(str(e))


def check_ssd_init(overlap, imsize):
    obj = StandardSSDInit()
    std = obj.get_standards()
    try:
        assert type(overlap) in std['overlap']['type'], "Invalid type for overlap_threshold argument. Please provide a {}".format(std['overlap']['type'])
        if type(imsize) is tuple:
            assert all(k == std['imsize']['type']
                       for k in imsize), "Invalid type for imsize argument. Please provide a {} type.".format(std['imsize']['type'])
        else:
            assert imsize == std['imsize']['type'], "Invalid type for imsize argument. Please provide a {} type.".format(
                std['imsize']['type'])
    except Exception as e:
        raise InvalidInputTypeError(str(e))

    try:
        assert overlap >= std['overlap']['range'][0] and overlap <= std['overlap']['range'][1], "Invalid value for overlap_threshold. Please provide a value between {} and {}.".format(std['overlap']['range'][0], std['overlap']['range'][1])
    except Exception as e:
        raise InvalidInputValueError(str(e))


def check_fcn_init(upscore):
    obj = StandardFCNInit()
    std = obj.get_standards()

    try:
        assert type(upscore) in std['upscore']['type'], "Invalid type for train_final_upscore argument. Please provide a {} type.".format(std['upscore']['type'])
    except Exception as e:
        raise InvalidInputTypeError(str(e))

def check_ternausnet_init(imsize):
    obj = StandardTernausNetInit()
    std = obj.get_standards()
    try:
        if type(imsize) is tuple:
            assert all(k % std['multiple'] == 0 for k in imsize), "Invalid imsize value. imsize for TernausNet must be integer multiple of {}.".format(
                std['multiple'])
        else:
            assert imsize % std['multiple'] == 0, "Invalid imsize value. imsize for TernausNet must be integer multiple of {}.".format(
                std['multiple'])
    except Exception as e:
        raise InvalidInputValueError(str(e))

def check_deeplabv3plus_init(imsize, scale_factor, atrous_rates, lr_initial, lr_power):
    obj = StandardDeeplabv3plusInit()
    std = obj.get_standards()
    try:
        assert type(scale_factor) in std['scale_factor']['type'], "Invalid type for scale_factor argument. Please provide a {}".format(std['scale_factor']['type'])
        assert type(atrous_rates) in std['atrous_rates']['type'], "Invalid type for atrous_rates argument. Please provide a {} type. You provided a {} type.".format(std['atrous_rates']['type'], type(atrous_rates))
        assert type(lr_initial) in std['lr_initial']['type'], "Invalid type for lr_initial argument. Please provide a {}".format(std['lr_initial']['type'])
        assert type(lr_power) in std['lr_power']['type'], "Invalid type for lr_power argument. Please provide a {}".format(std['lr_power']['type'])
    except Exception as e:
        raise InvalidInputTypeError(str(e))

    try:
        if type(imsize) is tuple:
            assert all(k in std['imsize']['range']
                       for k in imsize), "Invalid value for imsize argument. Please set imsize to ({},{}).".format(std['imsize']['range'][0], std['imsize']['range'][0])
        else:
            assert imsize in std['imsize']['range'], "Invalid value for imsize argument. Please set imsize to {}.".format(
                std['imsize']['range'])
        assert scale_factor in std['scale_factor']['value'], "Invalid value for scale_factor argument. Please set scale_factor to {}.".format(std['scale_factor']['value'])
        assert atrous_rates in std['atrous_rates']['value'], "Invalid value for atrous_rates argument. Please set atrous_rates to {}. You provided atrous_rates = {}".format(std['atrous_rates']['value'], atrous_rates)
        assert lr_initial >= std['lr_initial']['range'][0] and lr_initial <= std['lr_initial']['range'][1], "Invalid value for initial learning rate. Please provide a value between {} and {}.".format(std['lr_initial']['range'][0], std['lr_initial']['range'][1])
        assert lr_power >= std['lr_power']['range'][0] and lr_power <= std['lr_power']['range'][1], "Invalid value for learning rate power factor. Please provide a value between {} and {}.".format(std['lr_power']['range'][0], std['lr_power']['range'][1])
    except Exception as e:
        raise InvalidInputValueError(str(e))

def check_common_forward(x):
    obj = StandardForward()
    std = obj.get_standards()

    try:  # check for existance
        x
    except:
        raise MissingInputError(
            'Input argument x is missing in the forward function. Please call forward with valid input.')
    try:
        assert type(x) in std['type'], "Invalid type for input x. Please provide input with a {} type.".format(std['type'])
        assert len(x.shape) == std['length'], "Invalid number of dimensions for input x. Please provide input with len(x.shape) == {}".format(std['length'])
    except Exception as e:
        raise InvalidInputTypeError(str(e))

    try:
        assert x.dtype in std['value'], "Invalid value for input x. Please provide input with {}, {} or {} values.".format(std['value'][0], std['value'][1], std['value'][2])
    except Exception as e:
        raise InvalidInputValueError(str(e))


def check_yolov2_forward(anchor, x):
    obj = StandardYolov2Forward()
    std = obj.get_standards()
    check_common_forward(x)
    try:
        assert type(anchor) in std['anchor']['type'], "Invalid type for anchor argument. Please provide a {} type.".format(std['anchor']['type'])
    except Exception as e:
        raise InvalidInputTypeError(str(e))

    try:
        assert len(anchor) >= std['anchor']['range'][0] and len(
            anchor) <= std['anchor']['range'][1], "Invalid legnth for anchor argument. Please provide an object with length between {} and {}.".format(std['anchor']['range'][0], std['anchor']['range'][1])

        assert all(k.dtype in std['anchor']['value'] for k in anchor), "Invalid value for anchor argument. Please provide {} or {} values.".format(std['anchor']['value'][0], std['anchor']['value'][1])
    except Exception as e:
        raise InvalidInputValueError(str(e))


def check_common_learning_rate(lr):
    obj = StandardLR()
    std = obj.get_standards()
    try:
        assert type(lr) in std['LR']['type'],"Invalid type for learning rate argument. Please provide a {} type.".format(std['LR']['type'])
        assert lr >= std['LR']['range'][0] and lr < std['LR']['range'][1], "Invalid value for learning rate argument. Please provide values between {} and {}".format(std['LR']['range'][0], std['LR']['range'][1])
    except Exception as e:
        raise InvalidLearningRateError(str(e))

def check_missing_param(class_map):
    try:
        assert len(class_map)>0, "class_map not defined. Please define a class_map or load pretrained weights that contain a class_map."
    except Exception as e:
        raise MissingParamError(str(e))

def check_segmentation_label(label, n):
    if not np.sum(np.histogram(label, bins=list(range(256)))[0][n:-1]) == 0:
        raise InvalidInputValueError("Invalid label numbers in annotation data. Please provide annotation data with only numbers that correspond to the number of classes in class_map.")
    if not label.ndim == 2:
        raise InvalidInputTypeError("Invalid label data type with {} dimensions. Please provide label data with 2 dimensions only (label.ndim == 2).".format(label.ndim))

