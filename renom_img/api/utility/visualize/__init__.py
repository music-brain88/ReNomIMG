import numpy as np

from renom_img.api.classification.vgg import VGGBase, VGG16
from renom_img.api.classification.resnet import ResNetBase, ResNet50
from renom_img.api.classification.resnext import ResNeXtBase, ResNeXt50 
from renom.layers.activation.relu import Relu
from renom.core import UnaryOp, Node
from renom.debug_graph import showmark
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu

from renom.cuda import set_cuda_active


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
            dy = np.where(dy > 0, dy, 0)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy), **kwargs)


class Relu_gb:
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
    if isinstance(model, Sequential):
        model_dict = model.__dict__
        for k, v in model_dict.items():
            if isinstance(v, Relu):
                model_dict[k] = Relu_GB()
            elif k == '_layers':
                for i,e in enumerate(model_dict[k]):
                    if isinstance(e, Relu):
                        model_dict[k][i] = Relu_GB()
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


def vgg16_cam(model, x, class_id, mode, flag='cam'):
    y1 = model._model.block1(x)
    y2 = model._model.block2(y1)
    y3 = model._model.block3(y2)
    y4 = model._model.block4(y3)
    relu5_3 = rm.Sequential(model._model.block5[:-1])(y4)
    y5 = model._model.block5[-1](relu5_3)
    t = rm.flatten(y5)
    t = rm.relu(model._model.fc1(t))
    t = rm.relu(model._model.fc2(t))
    t = model._model.fc3(t)
    if mode == 'plus':
        t = rm.exp(t)
    t_c = t[:, class_id]
    if flag == 'cam':
        return rm.sum(t_c), relu5_3
    return rm.sum(t_c)


