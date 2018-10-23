# Before execution
#https://github.com/shelhamer/fcn.berkeleyvision.org
# You need to clone the above repository into ./external directory

from __future__ import print_function

import os
import os.path as osp
import sys

import numpy as np
try:
    import caffe
except ImportError:
    print('Cannot import caffe. Please install it.')
    quit(1)
import renom as rm

import fcn
import gdown

here = osp.dirname(osp.abspath(__file__))


sys.path.insert(0, osp.join(here, '../../fcn/external/fcn.berkeleyvision.org'))

def cached_download(url, path, md5=None, quiet=False):
    return gdown.download(url, path, quiet=False)

def caffe_to_renommodel(model, caffe_prototxt, caffemodel_path,
                          renommodel_path):
    #os.chdir(osp.dirname(caffe_prototxt))
    net = caffe.Net(caffe_prototxt, caffemodel_path, caffe.TEST)

    for name, param in net.params.items():
        try:
            layer = getattr(model._model, name)
        except AttributeError:
            print('Skipping caffe layer: %s' % name)
            sys.exit(1)
            continue

        has_bias = True
        if len(param) == 1:
            has_bias = False

        print('{0}:'.format(name))
        # weight
        if isinstance(layer, rm.Conv2d) or isinstance(layer, rm.Deconv2d):
            layer.params["w"] = rm.Variable(param[0].data[:,:,::-1,::-1])
        # bias
        if has_bias:
            layer.params["b"] = rm.Variable(param[1].data[np.newaxis,:,np.newaxis,np.newaxis])
    
    model.save(renommodel_path)


def main():
    class_map = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    for model_name in ['FCN8s']:
        print('[caffe_to_renommodel.py] converting model: %s' % model_name)
        model_name = model_name.lower()
        from renom_img.api.segmentation.fcn import FCN8s
        # get model
        model = FCN8s(np.arange(len(class_map)))

        # get caffemodel
        # detailed in https://github.com/shelhamer/fcn.berkeleyvision.org
        caffe_prototxt = osp.join(
            here, '.',
            'external/fcn.berkeleyvision.org/voc-%s/deploy.prototxt' %
            model_name)
        if not os.path.exists("./data/models/caffe"):
            os.makedirs("./data/models/caffe")
        caffemodel = "./data/models/caffe/%s-heavy-pascal.caffemodel" % model_name
        if not osp.exists(caffemodel):
            file = osp.join(osp.dirname(caffe_prototxt), "caffemodel-url")
            url = open(file).read().strip()
            cached_download(url, caffemodel)

        # convert caffemodel to renommodel
        renommodel = './data/models/renom/%s_from_caffe.h5' % model_name
        if not os.path.exists("./data/models/renom"):
            os.makedirs("./data/models/renom")
        if not osp.exists(renommodel):
            caffe_to_renommodel(model, caffe_prototxt,
                                  caffemodel, renommodel)


if __name__ == '__main__':
    main()
