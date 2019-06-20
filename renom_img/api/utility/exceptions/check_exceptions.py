import numpy as np
import renom as rm
from renom_img.api.utility.exceptions.standards import *
from renom_img.api.utility.exceptions.exceptions import *

def check_for_common_init_params(class_map,imsize,load_pretrained_weight,train_whole_network,target_model):

    standard_obj = StandardInit()
    # value check
    standards = standard_obj.get_standards()
    try:
        assert all(isinstance(k,standards['class_map']['value'][0]) for k in class_map),"Invalid value encounter in class_map. Type must be string.{}".format(class_map)
    
        if isinstance(imsize,tuple):
            assert all(isinstance(k,standards['imsize']['value'][0]) for k in imsize),"Invalid value encounter in imsize.{}".format(imsize)
        else:
            assert isinstance(imsize,standards['imsize']['value'][0]),"Invalid value encounter in imsize.{}".format(imsize)

        assert load_pretrained_weight in standards['load_pretrained_weight']['value'],"Invalid value passed for load_pretrained_weight.{}".format(load_pretrained_weight)

        assert train_whole_network in standards['train_whole_network']['value'],"Invalid value passed for train_whole_network.{}".format(train_whole_network)

    except Exception as e:
        raise ParamValueError(str(e))
    # type check
    try:
        assert type(class_map) in standards['class_map']['type'],"Type of class_map is invalid"
        assert type(imsize) in standards['imsize']['type'],"Type of imsize is invalid"
        assert type(load_pretrained_weight) in standards['load_pretrained_weight']['type'],"Type of load_pretrained_weight is invalid"
        assert type(train_whole_network) in standards['train_whole_network']['type'],"Type of train_whole_network is invalid"
        assert isinstance(target_model,standards['target_model']['type']),"Type of Model is not valid"
    except Exception as e:
        raise InvalidParamError(str(e))
    
    # check range limit
    try:
        assert len(class_map) >= standards['class_map']['range'][0] and len(class_map)<=standards['class_map']['range'][1],"length of class_map is invalid."
        if type(imsize) is tuple:
            assert imsize[0] >= standards['imsize']['range'][0] and imsize[0] <= standards['imsize']['range'][1] and imsize[1] >= standards['imsize']['range'][0] and imsize[1] <= standards['imsize']['range'][1],"imsize exceed the range."
        else:
            assert imsize >= standards['imsize']['range'][0] and imsize <= standards['imsize']['range'][1],"imsize exceeds the range."
    except Exception as e:
        raise InvalidParamError(str(e))


def check_resnet_init(plateau):
    standard_obj = StandardResNetInit()
    std = standard_obj.get_standards()
    try:
        assert plateau in std['plateau']['value'],"Invalid value for plateau."
    except Exception as e:
        raise ParamValueError(str(e))

    try:
        assert type(plateau) in std['plateau']['type'],"plateau type is invalid"
    except Exception as e:
        raise InvalidParamError(str(e))

def check_resnext_init(plateau,cardinality):
    standard_obj = StandardResNextInit()
    std = standard_obj.get_standards()
    try:
        assert plateau in std['plateau']['value'],"Invalid value for plateau"
    except Exception as e:
        raise ParamValueError(str(e))
    try:
        assert type(plateau) in std['plateau']['type'],"plateau type is invalid"
        assert type(cardinality) in std['cardinality']['type'],"cardinality type is invalid"

        assert cardinality >= std['cardinality']['range'][0] and cardinality <= std['cardinality']['range'][1],"Range limit for cardinality exceeded"
    except Exception as e:
        raise InvalidParamError(str(e))

def check_yolov1_init(cells,bbox):
    obj = StandardYolov1Init()
    std =obj.get_standards()

    try:
        assert type(cells) in std['cells']['type'],"Type of cells is invalid"
        assert type(bbox) in std['bbox']['type'],"Type of bbox is invalid"

        if type(cells) is tuple:
            assert cells[0] >= std['cells']['range'][0] and cells[0] <= std['cells']['range'][1] and cells[1] >= std['cells']['range'][0] and cells[1] <= std['cells']['range'][1],"Range limit exceed for cells"
        else:
            assert cells >= std['cells']['range'][0] and cells <= std['cells']['range'][1],"Range limit exceeds for cells"

        assert bbox >= std['bbox']['range'][0] and bbox <= std['bbox']['range'][1],"Range limit exceeds for bbox"
    except Exception as e:
        raise InvalidParamError(str(e))

def check_yolov2_init(imsize):
    obj = StandardYolov2Init()
    std = obj.get_standards()
    try:
        if type(imsize) is tuple:
            assert all(k%std['multiple']==0 for k in imsize),"Imsize for Yolov2 must be multiple of {}.".format(std['multiple'])
        else:
            assert imsize%std['multiple'] == 0,"Imsize for Yolov2 must be multiple of {}.".format(std['multiple'])
    except Exception as e:
        raise InvalidParamError(str(e))

def check_ssd_init(overlap,imsize):
    obj = StandardSSDInit()
    std = obj.get_standards()
    try:
        assert type(overlap) in std['overlap']['type'],"Type of overlap_threshold must be float."

        if type(imsize) is tuple:
            assert all(k==std['imsize']['type'] for k in imsize),"Imsize for SSD must be {}".format(std['imsize']['type'])
        else:
            assert imsize == std['imsize']['type'],"Imsize for SSD must be {}".format(std['imsize']['type'])
    
        assert overlap >= std['overlap']['range'][0] and overlap <= std['overlap']['range'][1],"Range limit for overlap_threshold exceeded."
    except Exception as e:
        raise InvalidParamError(str(e))


def check_fcn_init(upscore):
    obj = StandardFCNInit()
    std = obj.get_standards()

    try:
        assert upscore in std['upscore']['value'],"Value for train_final_upscore is invalid"
    except Exception as e:
        raise ParamValueError(str(e))

    try:
        assert type(upscore) in std['upscore']['type'],"Type of train_final_upscore is invalid"
    except Exception as e:
        raise InvalidParamError(str(e))

def check_common_forward(x):
    obj = StandardForward()
    std = obj.get_standards()

    try: # check for existance
        x
    except:
        raise MissingParamError('x is missing in the forward function. Please call forward with valid image data.')

    try:
        assert type(x) in std['type'],"type of x is invalid."
        assert len(x.shape)== std['length'],"shape of x is invalid in forward."
    except Exception as e:
        raise InvalidParamError(str(e))

    try:
        assert x.dtype in std['value'],"value of x is invalid."
    except Exception as e:
        raise ParamValueError(str(e))

def check_yolov2_forward(anchor,x):
    obj = StandardYolov2Forward()
    std = obj.get_standards()
    check_common_forward(x)
    try:
        assert type(anchor) in std['anchor']['type'],"anchor type is invalid."
        assert len(anchor)>= std['anchor']['range'][0] and len(anchor) <=std['anchor']['range'][1],"number of anchor is invalid"
    except Exception as e:
        raise InvalidParamError(str(e))

    try:
        assert all(k.dtype in std['anchor']['value'] for k in anchor),"anchor value is invalid"
    except Exception as e:
        raise ParamValueError(str(e))

