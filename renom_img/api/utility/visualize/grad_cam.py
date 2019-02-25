from renom.layers.activation.relu import Relu
from renom_img.api.utility.visualize import model_types, Relu_gb, convert_relus
from renom_img.api.utility.visualize.tools import load_img, preprocess_img, visualize_grad_cam, visualize_comparison
from renom_img.api.utility.visualize import vgg16_cam, resnet50_cam, resnext50_cam

class Guided_Grad_Cam():

    def __init__(self, model_cam):
        # define all attributes
        self.model_cam = model_cam
        self._model_type = self.get_model_type(self.model_cam)
        assert self._model_type in model_types, "Model must be instance of {}".format(model_types)
        self.model_gb = self.convert_relus(model_cam)
        self.model_cam.set_models(inference=True)
        self.model_gb.set_models(inference=True)


    def get_model_type(self, model_cam):
        return model_cam.__class__.__name__


    def convert_relus(self, model):
        model_dict = model.__dict__
        for k, v in model_dict.items():
            if isinstance(v, Relu):
                model_dict[k] = Relu_gb()
            else:
                try:
                    x = model_dict[k].__dict__
                    if '_parameters' not in x.keys():
                        convert_relus(x)
                except:
                    if isinstance(v, list):
                        for e in v:
                            convert_relus(e)
        return model


    # 1a. Forward pass (Grad-CAM)
    def forward_cam(self, x, class_id, mode):
        if (self._model_type == 'VGG16'):
            y_c, final_conv = vgg16_cam(self.model_cam, x, class_id, mode, flag='cam')
        elif (self._model_type == 'ResNet50'):
            y_c, final_conv = resnet50_cam(self.model_cam, x, class_id, mode, flag='cam')
        elif (self._model_type == 'ResNeXt50'):
            y_c, final_conv = resnext50_cam(self.model_cam, x, class_id, mode, flag='cam')
        elif (self._model_type == 'Sequential'):
            y_c, final_conv = sequential_cam(self.model_cam, x, class_id, mode, flag='cam', node=None)
            # user-defined custom model
            pass
        else:
            # error statement
            pass
        return y_c, final_conv


    # 1b. Forward pass (Guided Backpropagation)
    def forward_gb(self, x, class_id, mode):
        # check model type
        if (self._model_type = 'VGG16'):
            y_gb = vgg16_cam(self.model_gb, x, class_id, mode, flag='gb')
        elif (self._model_type = 'ResNet50'):
            y_gb = resnet50_cam(self.model_gb, x, class_id, mode, flag='gb')
        elif (self._model_type = 'ResNeXt50'):
            y_gb = resnext50_cam(self.model_gb, x, class_id, mode, flag='gb')
        elif (self._model_type = sequential):
            # user-defined custom model
            y_gb = sequential_cam(self.model_gb, x, class_id, mode, flag='gb') 
        else:
            # error message
            pass
        return y_gb


    def get_predicted_class(self, x):
        return np.argmax(self.model_cam(x))


    def guided_backprop(self, x_gb, y_gb):
        grad_gb = y_gb.grad()
        input_map = np.squeeze(grad_gb.get(x_gb)) #ok
        input_map = input_map[::-1,:,:] #ok
        input_map = input_map.transpose(1,2,0)
        gb_viz = input_map.copy()
        input_map -= np.min(gb_viz)
        input_map /= np.max(input_map)

        return gb_viz, input_map


    def generate_map(self, y_c, final_conv, mode)
    # 3a. Grad-CAM (coefficients)
        grad = y_c.grad()
        A = np.squeeze(final_conv)
        dAk = np.squeeze(grad.get(final_conv))

        if mode='plus':
            alpha_new = (dAk * dAk)
            denominator_1 = 2*dAk*dAk
            denominator_2 = rm.sum(rm.sum(A, axis=2), axis=1)
            denominator_2 = denominator_2[:, np.newaxis, np.newaxis]
            denominator_2 *= dAk*dAk*dAk
            alpha_new = alpha_new / (denominator_1 + denominator_2 + 1e-8)
            w = rm.sum(rm.sum(alpha_new*rm.relu(dAk), axis=2), axis=1)
            w = w[:, np.newaxis, np.newaxis]  
        else:
            w = rm.sum(rm.sum(dAk, axis=2), axis=1) / (dAk.shape[1]*dAk.shape[2])
            w = w[:,np.newaxis,np.newaxis]

        # 3b. Grad-CAM (saliency map)
        L = rm.relu(rm.sum(w * A, axis=0))
        #TODO: zoom_factor, not hard-coding
        # L = scipy.ndimage.zoom(L, zoom_factor, order=1)
        L = scipy.ndimage.zoom(L, 16, order=1)
        L_big = L[:, :, np.newaxis]

        # 4. Guided Grad-CAM
        result = L_big * gb_viz
        result -= np.min(result)
        result /= np.max(result)

        return L, result


    # function to actually run calculation
    def __call__(img_path, size=(224,224), class_id=None, mode='normal'):
        # move preprocessing over to user-side (before calling this)
        #img = load_img(img_path, size)
        #x = preprocess_img(img)
        x_gb = x.copy()

        if not class_id:
            class_id = self.get_predicted_class(x)

        y, final_conv = self.forward_cam(x, class_id, mode)

        y_gb = self.forward_gb(x_gb, class_id, mode)
        gb_map, input_map = self.guided_backprop(x_gb, y_gb)
        L, result = self.generate_map(y, final_conv, gb_map, mode) 

        return img, input_map, L, result
