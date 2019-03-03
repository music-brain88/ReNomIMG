import numpy as np
import renom as rm
import renom
from renom_img.api.classification.vgg import VGGBase, VGG16
from renom_img.api.classification.resnet import ResNetBase, ResNet50
from renom_img.api.classification.resnext import ResNeXtBase, ResNeXt50 
from renom.layers.activation.relu import Relu
from renom.layers.function.parameterized import Sequential
from renom.core import UnaryOp, Node
from renom.debug_graph import showmark
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu

from renom.cuda import set_cuda_active


model_types = ['VGG16', 'VGG19', 'ResNet50', 'ResNeXt50']

#Guided Back-propagation version of ReLU function
@showmark
class relu_gb(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.maximum(arg, 0)

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cu.curelu_foward(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dy = np.where(dy > 0, dy, 0)
            self.attrs._arg._update_diff(context, np.where(self == 0, 0, dy), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = get_gpu(self.attrs._arg).empty_like_me()
            cu.curelu_backard(get_gpu(self.attrs._arg), dx)
            dy = dy.as_ndarray()
            dy = np.where(dy > 0, dy, 0)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy), **kwargs)


class Relu_GB:
    '''Modified Rectified Linear Unit activation function for Guided Backpropagation.
       Backward pass is modified according to reference below

        :math:`f(x)=max(x, 0)`

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> x = np.array([[1, -1]])
        array([[ 1, -1]])
        >>> rm.relu(x)
        relu([[ 1.  , 0.]])

        >>> # instantiation
        >>> activation = rm.Relu()
        >>> activation(x)
        relu([[ 1.  , 0]])

    '''

    def __call__(self, x):
        return relu_gb(x)


def convert_relus(model):
    if isinstance(model, Relu):
        model = Relu_GB()
    elif isinstance(model, Sequential):
        model_dict = model.__dict__
        for k, v in model_dict.items():
            if isinstance(v, Relu):
                model_dict[k] = Relu_GB()
            elif k == '_layers':
                for i,e in enumerate(model_dict[k]):
                    if isinstance(e, Relu):
                        model_dict[k][i] = Relu_GB()
                    else:
                        convert_relus(model_dict[k][i])
            else:
                convert_relus(model_dict[k])
    else:
        try:
            model_dict = model.__dict__
            for k, v in model_dict.items():
                if isinstance(v, Relu):
                    model_dict[k] = Relu_GB()
                elif '_parameters' not in k:
                    convert_relus(model_dict[k])
        except:
            if isinstance(model, list):
                for e in model:
                    if isinstance(e, Relu):
                        model[e] = Relu_GB()
                    else:
                        convert_relus(model[e])
    return model


def vgg_cam(model, x, class_id, mode):
    x = model._model.block1(x)
    x = model._model.block2(x)
    x = model._model.block3(x)
    x = model._model.block4(x)
    final_conv = rm.Sequential(model._model.block5[:-1])(x)
    x = model._model.block5[-1](final_conv)
    x = rm.flatten(x)
    x = rm.relu(model._model.fc1(x))
    x = rm.relu(model._model.fc2(x))
    x = model._model.fc3(x)
    if mode == 'plus':
        x = rm.exp(x)
    x_c = x[:, class_id]
    return rm.sum(x_c), final_conv


def resnet_cam(model, x, class_id, mode):
    x = model._model.conv1(x)
    x = model._model.bn1(x)
    x = model._model.relu(x)
    x = model._model.maxpool(x)
    x = model._model.layer1(x)
    x = model._model.layer2(x)
    x = model._model.layer3(x)
    final_conv = model._model.layer4(x)
    x = rm.average_pool2d(final_conv, filter=(final_conv.shape[2],final_conv.shape[3]))
    x = model._model.flat(x)
    x = model._model.fc(x)
    if mode == 'plus':
        x = rm.exp(x)
    x_c = x[:, class_id]
    return rm.sum(x_c), final_conv


def sequential_cam(model, x, class_id, mode, node):
    pass
