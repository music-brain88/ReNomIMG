import os
import os.path as osp
import sys
import numpy as np
try:
    import caffe
except ImportError:
    print("Cannot import caffe.")
    quit(1)
import gdown
import renom as rm

def caffe_to_renom(model, caffe_prototxt, caffemodel, renommodel):
    net = caffe.Net(caffe_prototxt, caffemodel, caffe.TEST)
    for name, param in net.params.items():
        if name[:2] == "fc":
            print("SKIP:{}".format(name))
            continue
        try:
            layer = getattr(model._model, name)
        except AttributeError:
            sys.exit(1)
        has_bias = True
        if len(param) == 1:
            has_bias = False
        print("{}:".format(name))
        if isinstance(layer, rm.Conv2d) or isinstance(layer, rm.Deconv2d):
            layer.params["w"] = rm.Variable(param[0].data[:,:,::-1,::-1])
            if has_bias:
                layer.params["b"] = rm.Variable(param[1].data[np.newaxis, :, np.newaxis])
        model.save(renommodel)

def cached_download(url, path):
    return gdown.download(url, path, quiet=False)

def main():
    from renom_img.api.classification.vgg import VGG16_NODENSE
    model = VGG16_NODENSE()
    caffe_prototxt = "./external/VGG_ILSVRC_16/VGG_ILSVRC_16_layers_deploy.prototxt"
    if not osp.exists(caffe_prototxt):
        if not osp.exists("./external/VGG_ILSVRC_16"):
            os.makedirs("./external/VGG_ILSVRC_16")
        cached_download("https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt",caffe_prototxt)
    caffemodel = "./data/models/caffe/VGG_ILSVRC_16_layers.caffemodel"
    if not osp.exists(caffemodel):
        if not osp.exists("./data/models/caffe"):
            os.makedirs("./data/models/caffe")
        cached_download("http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel", caffemodel)
    renommodel = "./data/models/renom/vgg16_from_caffe.h5"
    if not osp.exists(renommodel):
        if not osp.exists("./data/models/renom"):
            os.makedirs("./data/models/renom")
    caffe_to_renom(model, caffe_prototxt, caffemodel, renommodel)
    
if __name__ == "__main__":
    main()
