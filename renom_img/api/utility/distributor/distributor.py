import os
import threading
import numpy as np
from PIL import Image
from queue import Queue
from concurrent.futures import ThreadPoolExecutor as Executor

from renom_img.api.utility.load import load_img


class ImageDistributorBase(object):

    def __init__(self, img_path_list, label_list=None,
                 target_builder=None,
                 augmentation=None, num_worker=4):
        self._img_path_list = img_path_list
        self._label_list = label_list
        self._num_worker = num_worker
        self._augmentation = augmentation
        self._builder = target_builder

    def __len__(self):
        return len(self._img_path_list)

    @property
    def img_path_list(self):
        return self._img_path_list

    @property
    def annotation_list(self):
        return self._label_list

    def get_resized_annotation_list(self, imsize):
        """This function only for object detection.
        """
        resized_annotation_list = []
        for path, annotation in zip(self.img_path_list, self.annotation_list):
            ims = Image.open(path).size
            sw = 1. / ims[0]
            sh = 1. / ims[1]
            resized_annotation_list.append([{
                'box': [obj['box'][0] * sw, obj['box'][1] * sh, obj['box'][2] * sw, obj['box'][3] * sh],
                'class': obj['class'],
                'name': obj['name']}
                for obj in annotation])
        return resized_annotation_list

    def batch(self, batch_size, callback=None, shuffle=True):
        """
        Default: Classification
            Detection
            Segmentation
        Input data format is specified with task.
        """
        N = len(self)
        batch_loop = int(np.ceil(N / batch_size))
        builder = callback
        if builder is None:
            builder = self._builder
        if builder is None:
            def builder(x, y, aug): return aug(np.vstack([load_img(path) for path in x]), y)

        if shuffle:
            if N < 100000:
                perm = np.random.permutation(N)
            else:
                perm = np.random.randint(0, N, size=(N, ))
        else:
            perm = np.arange(N)

        def build(args):
            img_path_list, annotation_list = args
            # Otherwise, callback owns what transformation will be performed.
            return builder(img_path_list, annotation_list, self._augmentation)

        with Executor(max_workers=self._num_worker) as exector:
            batch_perm = [perm[nth * batch_size:(nth + 1) * batch_size]
                          for nth in range(batch_loop)]
            arg = [
                ([self._img_path_list[p] for p in bp], [self._label_list[p] for p in bp])
                for bp in batch_perm
            ]
            # for result in exector.map(build, arg):
            #     yield result
            yield from exector.map(build, arg)


class ImageDistributor(ImageDistributorBase):

    def __init__(self, img_path_list, label_list=None,
                 target_builder=None,
                 augmentation=None, num_worker=4):
        super(ImageDistributor, self).__init__(img_path_list,
                                               label_list, target_builder, augmentation, num_worker)

    def batch(self, batch_size, target_builder=None, shuffle=True):
        return super(ImageDistributor, self).batch(batch_size, target_builder, shuffle)
