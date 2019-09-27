import copy
import numpy as np
from scipy.ndimage import zoom
import renom as rm
from renom.layers.activation.relu import Relu
from renom.cuda import is_cuda_active, set_cuda_active
from renom_img.api.utility.visualize import model_types, Relu_GB, convert_relus
from renom_img.api.utility.visualize.tools import visualize_grad_cam, visualize_comparison
from renom_img.api.utility.visualize import vgg_cam, resnet_cam, sequential_cam


class GuidedGradCam():
    """ Guided Grad-cam implementation for visualizing CNN classification model feature map importance

    Args:
        model_cam (ReNom model instance): CNN-based classification model to be used
                                          for creating Guided Grad-CAM saliency maps.
                                          Model must be ReNom instance of VGG, ResNet,
                                          ResNeXt or rm.Sequential. Model should use
                                          ReLu activation functions and be pre-trained
                                          on the same dataset used for Grad-CAM visualizations.

    Returns:
        (numpy.ndarray): Guided backpropagation array, Grad-CAM(++) saliency map array, Guided Grad-CAM(++) array

    Example:
        >>> #This sample uses matplotlib to display files, so it is recommended to run this inside a Jupyter Notebook
        >>>
        >>> import renom as rm
        >>> import numpy as np
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.pyplot import cm
        >>> from renom_img.api.classification.vgg import VGG16
        >>> from renom_img.api.utility.visualize.grad_cam import GuidedGradCam
        >>> from renom_img.api.utility.visualize.tools import load_img, preprocess_img, visualize_grad_cam
        >>>
        >>> model = VGG16()
        >>>
        >>> #Provide pre-trained model weights for same dataset you are producing Grad-CAM visualizations on
        >>> model.load("my_pretrained_weights.h5")
        >>>
        >>> #Create Grad-CAM instance based on pre-trained model (VGG, ResNet, ResNeXt, or rm.Sequential)
        >>> grad_cam = GuidedGradCam(model)
        >>>
        >>> #Provide path to image file for producing Grad-CAM visualizations
        >>> img_path = '/home/username/path/to/images/cat_dog.jpg'
        >>>
        >>> #Load and pre-process image (must be same pre-processing as used during training)
        >>> img = Image.open(img_path)
        >>> size=(224,224)
        >>> img = load_img(img_path, size)
        >>> x = preprocess_img(img)
        >>>
        >>> #Select class_id (index of array in model's final output) to produce visualizations for. Must be consistent with class ID in trained model.
        >>> class_id = 243
        >>>
        >>> #Generate Grad-CAM maps
        >>> input_map, L, result = grad_cam(x, size, class_id=class_id, mode='normal')
        >>>
        >>> #Overlay Grad-CAM saliency map on image using matplotlib
        >>> plt.imshow(img)
        >>> plt.imshow(L, cmap = cm.jet, alpha = 0.6)
        >>> plt.axis("off")
        >>> plt.savefig("grad_cam_sample.png", bbox_inches='tight', pad_inches=0)
        >>>
        >>> #Visualize Guided Grad-CAM (original image, guided backpropagation, Grad-CAM saliency map, Guided Grad-CAM visualization)
        >>> visualize_grad_cam(img, input_map, L, result)
        >>>
        >>> #Generate Grad-CAM++ maps
        >>> input_map, L, result = grad_cam(x, size, class_id=class_id, mode='plus')
        >>> #Visualize results (original image, guided backpropagation, Grad-CAM++ saliency map, Guided Grad-CAM++ visualization)
        >>> visualize_grad_cam(img, input_map, L, result)

    References:
        | Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra
        | **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**
        | https://arxiv.org/abs/1610.02391 
        |
        | Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N Balasubramanian
        | **Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks**
        | https://arxiv.org/abs/1710.11063
        |

    """

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
        """ Gets model type information for model passed to Grad-CAM 

        Args:
            model_cam (ReNom model instance): CNN-based classification model to be used
                                          for creating Guided Grad-CAM saliency maps
                                          Model must be ReNom instance of VGG, ResNet,
                                          ResNeXt or rm.Sequential. Model should use
                                          ReLu activation functions and be pre-trained
                                          on the same dataset used for Grad-CAM visualizations.

        Returns:
            (string): Model class name
        """
        return model_cam.__class__.__name__

    def check_for_relus(self, model):
        """ Assertion check to see if ReLu activation functions exist in model

        Args:
            model_cam (ReNom model instance): CNN-based classification model to be used
                                          for creating Guided Grad-CAM saliency maps
                                          Model must be ReNom instance of VGG, ResNet,
                                          ResNeXt or rm.Sequential. Model should use
                                          ReLu activation functions and be pre-trained
                                          on the same dataset used for Grad-CAM visualizations.

        Returns:
            (bool): assert result
        """
        assert any('Relu' in i for i in map(str, (i.__class__.__name__ for i in model._layers))), \
            'Model must contain at least one Relu'

    def get_scaling_factor(self, size, L):
        """ Calculates scaling factor for aligning Grad-CAM map and input image sizes

        Args:
            size (tuple): tuple of integers representing original image size
            L (ndarray): Grad-CAM saliency map

        Returns:
            float, float: width and height scaling factors for aligning final array sizes
        """
        return float(size[1] / L.shape[0]), float(size[0] / L.shape[1])

    # 1a. Forward pass (Grad-CAM)
    def forward_cam(self, x, class_id, mode, node):
        """ Calculates forward pass through model for Grad-CAM

        Args:
            x (renom.Variable): Input data for model after pre-processing has been applied
            class_id (int): Class ID for creating visualizations
            mode (string): Flag for selecting Grad-CAM or Grad-CAM++
            node (int): Index representing final convolutional layer (used in rm.Sequential case only)

        Returns:
            (renom.Variable): Final layer output and final convolution layer output
        """
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
        """ Calculates forward pass through model for guided backpropagation

        Args:
            x_gb (renom.Variable): Input data for model after pre-processing has been applied
            class_id (int): Class ID for creating visualizations
            mode (string): Flag for selecting Grad-CAM or Grad-CAM++

        Returns:
            (renom.Variable): Final layer output
        """
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
        """ Returns class that model predicts given input data

        Args:
            x (renom.Variable): Input data for model after pre-processing has been applied
            class_id (int): Class ID for creating visualizations
            mode (string): Flag for selecting Grad-CAM ('normal', default) or Grad-CAM++ ('plus')

        Returns:
            (int): np.argmax index of final model output
        """
        if is_cuda_active():
            t = self.model_cam(x).as_ndarray()
        else:
            t = self.model_cam(x)
        return np.argmax(t)

    def guided_backprop(self, x_gb, y_gb):
        """ Calculates guided backpropagation backward pass

        Args:
            x_gb (renom.Variable): Input data for model after pre-processing has been applied
            y_gb (renom.Variable): Output of guided backpropagaion forward pass for x_gb

        Returns:
            (numpy.ndarray): Raw and normalized guided backpropagation outputs
        """
        if is_cuda_active:
            y_gb.to_cpu()
        grad = y_gb.grad()
        if is_cuda_active():
            grad_gb = grad.get(x_gb).as_ndarray()
        else:
            grad_gb = grad.get(x_gb)
        input_map = np.squeeze(grad_gb, axis=0)
        input_map = input_map.transpose(1, 2, 0)
        gb_viz = input_map.copy()
        input_map -= np.min(input_map)
        if np.max(input_map) == 0:
            input_map /= (np.max(input_map) + 1e-8)
        else:
            input_map /= np.max(input_map)
        return gb_viz, input_map

    def generate_map(self, y_c, final_conv, gb_map, mode, size):
        """ Generates Guided Grad-CAM and Grad-CAM saliency maps as numpy arrays

        Args:
            y_c (renom.Variable): Output of final layer in forward pass through model
            final_conv (renom.Variable): Output of final convolution layer in forward pass through model
            gb_map (numpy.ndarray): numpy array representing normalized guided backpropagation output
            mode (string): Flag for selecting Grad-CAM ('normal', default) or Grad-CAM++ ('plus')

        Returns:
            (numpy.ndarray): Grad-CAM saliency map and Guided Grad-CAM map as numpy arrays
        """
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
        scale_w, scale_h = self.get_scaling_factor(size, L)
        L = zoom(L, (scale_w, scale_h), order=1)
        L_big = np.expand_dims(L, 2)
        # 4. Guided Grad-CAM
        result = L_big * gb_map
        if result.shape[2] == 1:
            result = np.squeeze(result, axis=2)
        result -= np.min(result)
        if np.max(result) == 0:
            result /= (np.max(result) + 1e-8)
        else:
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
