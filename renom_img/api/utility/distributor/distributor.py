import os
import threading
import numpy as np
from PIL import Image
from queue import Queue
from concurrent.futures import ProcessPoolExecutor as Executor

from renom_img.api.utility.load import load_img


class BuilderWrapper():

    def __init__(self, builder, augmentation):
        self.builder = builder
        self.augmentation = augmentation

    def __call__(self, args):
        img_path_list, annotation_list, nth = args
        return self.builder(img_path_list, annotation_list, augmentation=self.augmentation, nth=nth)


class ImageDistributorBase(object):
    """Base class distribute images.

    """

    def __init__(self, img_path_list, label_list=None,
                 target_builder=None,
                 augmentation=None,
                 imsize=None,
                 num_worker=3):
        self._img_path_list = img_path_list
        self._label_list = label_list
        self._num_worker = num_worker
        self._augmentation = augmentation
        self._builder = target_builder
        self._imsize = imsize

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

        Args:
            imsize(float, float): size of image

        Return:
              resized_annotation_list(list): resized_annotation
        """
        resized_annotation_list = []
        for path, annotation in zip(self.img_path_list, self.annotation_list):
            ims = Image.open(path).size
            sw = 1. / ims[0]
            sh = 1. / ims[1]
            resized_annotation_list.append([{
                'box': [obj['box'][0] * sw, obj['box'][1] * sh, obj['box'][2] * sw, obj['box'][3] * sh],
                **{k: v for k, v in obj.items() if k != 'box'}
            }
                for obj in annotation])
        return resized_annotation_list

    def batch(self, batch_size, callback=None, shuffle=True):
        """

        Default

            * Classification
            * Detection
            * Segmentation

        Input data format is specified with task.
        """
        N = len(self)
        batch_loop = int(np.ceil(N / batch_size))
        builder = callback

        # User can override builder giving 'callback' to this method.
        if builder is None:
            builder = self._builder

        if builder is None:
            builder = lambda x, y, **kwargs: (x, y)

        if shuffle:
            if N < 100000:
                perm = np.random.permutation(N)
            else:
                perm = np.random.randint(0, N, size=(N, ))
        else:
            perm = np.arange(N)

        with Executor(max_workers=self._num_worker) as exector:
            batch_perm = [perm[nth * batch_size:(nth + 1) * batch_size]
                          for nth in range(batch_loop)]
            if self._label_list is None:
                arg = [([self._img_path_list[p] for p in bp], None, i)
                       for i, bp in enumerate(batch_perm)]
            else:
                arg = [
                    ([self._img_path_list[p] for p in bp],
                     [self._label_list[p] for p in bp], i)
                    for i, bp in enumerate(batch_perm)]

            # For avoiding memory over flow.
            # Don't submit all thread at once.
            iter_count = 0
            work_thread = []
            while (iter_count - len(work_thread)) < len(arg):
                for i in range(min(self._num_worker * 4 - len(work_thread), len(arg) - iter_count)):
                    work_thread.append(exector.submit(BuilderWrapper(
                        builder, self._augmentation), arg[iter_count]))
                    iter_count += 1
                yield work_thread.pop(0).result()


class ImageDistributor(ImageDistributorBase):

    def __init__(self, img_path_list, label_list=None,
                 target_builder=None,
                 augmentation=None,
                 imsize=None,
                 num_worker=3):
        super(ImageDistributor, self).__init__(img_path_list,
                                               label_list, target_builder, augmentation, imsize, num_worker)

    def batch(self, batch_size, target_builder=None, shuffle=True):
        """

        Args:
            batch_size(int): batch size
            target_builder(ImageDistributor): target builder
            shuffle(bool): shuffle or not when splitting data

        Yields:
            (path of images(list), path of labels(list)

       """
        return super(ImageDistributor, self).batch(batch_size, target_builder, shuffle)

    def split(self, ratio, shuffle=True):
        """ split image and laebls

        Args:
            ratio(float): ratio between training set and validation set
            shuffle(bool): shuffle or not when splitting data
        """
        assert ratio < 1.0 and ratio > 0.0
        data1_N = int(ratio * len(self))
        if shuffle:
            perm = np.random.permutation(len(self))
            perm1 = perm[:data1_N]
            perm2 = perm[data1_N:]
            return ImageDistributor([self.img_path_list[p] for p in perm1], [self.annotation[p] for p in perm1], self._builder, self._augmentation, self._imsize, self._num_worker), \
                ImageDistributor([self.img_path_list[p] for p in perm2], [self.annotation[p]
                                                                          for p in perm2], self._builder, self._augmentation, self._imsize, self._num_worker)
        else:
            perm = np.arange(len(self))
            perm1 = perm[:data1_N]
            perm2 = perm[data1_N:]
            return ImageDistributor([self.img_path_list[p] for p in perm1], [self.annotation[p] for p in perm1], self._builder, self._augmentation, self._imsize, self._num_worker), \
                ImageDistributor([self.img_path_list[p] for p in perm2], [self.annotation[p]
                                                                          for p in perm2], self._builder, self._augmentation, self._imsize, self._num_worker)
