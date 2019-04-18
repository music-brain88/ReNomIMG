import copy
import numpy as np
from scipy.ndimage import zoom
import renom as rm
from renom.layers.activation.relu import Relu
from renom.cuda import is_cuda_active, set_cuda_active
from renom_img.api.utility.visualize import model_types, Relu_GB, convert_relus
from renom_img.api.utility.visualize.tools import visualize_grad_cam, visualize_comparison
from renom_img.api.utility.visualize import vgg_cam, resnet_cam, sequential_cam


class Guided_Grad_Cam():

    def __init__(self, model_cam):
        self.model_cam = model_cam
        self._model_type = self.get_model_type(self.model_cam)
        assert self._model_type in model_types, "Model must be instance of {}".format(model_types)
        if self._model_type == 'Sequential':
            self.check_for_relus(self.model_cam)
        self.model_gb = convert_relus(copy.deepcopy(self.model_cam))
        self.model_cam.set_models(inference=True)
        self.model_gb.set_models(inference=True)

    def get_model_type(self, model_cam):
        return model_cam.__class__.__name__

    def check_for_relus(self, model):
        assert any('Relu' in i for i in map(str, (i.__class__.__name__ for i in model._layers))), \
            'Model must contain at least one Relu'

    def get_zoom_factor(self, size, L):
        return int(size[0] / L.shape[0])

    # 1a. Forward pass (Grad-CAM)

    def forward_cam(self, x, class_id, mode, node):
        if 'VGG' in self._model_type:
            y_c, final_conv = vgg_cam(self.model_cam, x, class_id, mode)
        elif 'ResNet' in self._model_type or 'ResNeXt' in self._model_type:
            y_c, final_conv = resnet_cam(self.model_cam, x, class_id, mode)
        elif self._model_type == 'Sequential':
            y_c, final_conv = sequential_cam(self.model_cam, x, class_id, mode, node)
        else:
            print("Error: Model must be of type VGG, ResNet, ResNeXt or rm.Sequential")
        return y_c, final_conv

    # 1b. Forward pass (Guided Backpropagation)

    def forward_gb(self, x_gb, class_id, mode):
        t_gb = self.model_gb(x_gb)
        if mode == 'plus':
            t_gb = rm.exp(t_gb)
        if t_gb.shape[1] > 1:
            t_gb_c = t_gb[:, class_id]
        else:
            t_gb_c = t_gb
        y_gb = rm.sum(t_gb_c)
        return y_gb

    def get_predicted_class(self, x):
        if is_cuda_active():
            t = self.model_cam(x).as_ndarray()
        else:
            t = self.model_cam(x)
        return np.argmax(t)

    def guided_backprop(self, x_gb, y_gb):
        if is_cuda_active:
            y_gb.to_cpu()
        grad = y_gb.grad()
        if is_cuda_active():
            grad_gb = grad.get(x_gb).as_ndarray()
        else:
            grad_gb = grad.get(x_gb)
        input_map = np.squeeze(grad_gb, axis=0)
        #input_map = input_map[::-1,:,:]
        input_map = input_map.transpose(1, 2, 0)
        gb_viz = input_map.copy()
        input_map -= np.min(input_map)
        input_map /= np.max(input_map)
        return gb_viz, input_map

    def generate_map(self, y_c, final_conv, gb_map, mode, size):
        # 3a. Grad-CAM (coefficients)
        grad = y_c.grad()
        if is_cuda_active():
            A = np.squeeze(final_conv.as_ndarray())
            dAk = np.squeeze(grad.get(final_conv).as_ndarray())
        else:
            A = np.squeeze(final_conv)
            dAk = np.squeeze(grad.get(final_conv))
        if mode == 'plus':
            alpha_new = (dAk * dAk)
            term_1 = 2 * dAk * dAk
            term_2 = rm.sum(rm.sum(A, axis=2), axis=1)
            if is_cuda_active():
                term_2 = term_2.as_ndarray()
            term_2 = term_2[:, np.newaxis, np.newaxis]
            term_2 = term_2 * dAk * dAk * dAk
            alpha_new = alpha_new / (term_1 + term_2 + 1e-8)
            w = rm.sum(rm.sum(alpha_new * rm.relu(dAk), axis=2), axis=1)
            if is_cuda_active():
                w = w.as_ndarray()
            w = w[:, np.newaxis, np.newaxis]
        else:
            w = rm.sum(rm.sum(dAk, axis=2), axis=1) / (dAk.shape[1] * dAk.shape[2])
            if is_cuda_active():
                w = w.as_ndarray()
            w = w[:, np.newaxis, np.newaxis]

        # 3b. Grad-CAM (saliency map)
        L = rm.relu(rm.sum(A * w, axis=0)).as_ndarray()
        zoom_factor = self.get_zoom_factor(size, L)
        L = zoom(L, zoom_factor, order=1)
        L_big = np.expand_dims(L, 2)
        # 4. Guided Grad-CAM
        result = L_big * gb_map
        if result.shape[2] == 1:
            result = np.squeeze(result, axis=2)
        result -= np.min(result)
        result /= np.max(result)

        return L, result

    # function to actually run calculation
    def __call__(self, x, size=(224, 224), class_id=None, mode='normal', node=None):
        x_gb = rm.Variable(x.copy())
        x = rm.Variable(x)

        if not class_id:
            class_id = self.get_predicted_class(x)

        y, final_conv = self.forward_cam(x, class_id, mode, node)

        y_gb = self.forward_gb(x_gb, class_id, mode)
        gb_map, input_map = self.guided_backprop(x_gb, y_gb)
        L, result = self.generate_map(y, final_conv, gb_map, mode, size)

        return input_map, L, result
