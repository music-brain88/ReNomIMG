import os
import threading
import numpy as np
from PIL import Image
from queue import Queue
from concurrent.futures import ThreadPoolExecutor as Executor

from renom_img.api.utility.target import DataBuilderClassification


RESIZE_METHOD = Image.BILINEAR

class ImageDistributorBase(object):

    def __init__(self, img_path_list, label_list=None,
            target_builder=DataBuilderClassification((224, 224)),
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


    def batch(self, batch_size, callback=None, shuffle=True):
        """
        Default: Classification
            Detection
            Segmentation
        Input data format is specified with task.
        """
        N = len(self)
        batch_loop = int(np.ceil(N/batch_size))
        builder = callback
        if builder is None:
            builder = self._builder

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
            batch_perm = [perm[nth*batch_size:(nth+1)*batch_size] for nth in range(batch_loop)]
            arg = [
                ([self._img_path_list[p] for p in  bp], [self._label_list[p] for p in bp])
                for bp in batch_perm
            ]
            # for result in exector.map(build, arg):
            #     yield result
            yield from exector.map(build, arg)


class ImageDistributor(ImageDistributorBase):

    def __init__(self, img_path_list, label_list=None,
            target_builder=DataBuilderClassification((224, 224)),
            augmentation=None, num_worker=4):
        super(ImageDistributor, self).__init__(img_path_list,
                                               label_list, target_builder, augmentation, num_worker)

    def batch(self, batch_size, target_builder=None, shuffle=True):
        return super(ImageDistributor, self).batch(batch_size, target_builder, shuffle)


if __name__ == "__main__":
    import os
    import time
    from renom_img.api.utility.load import parse_xml_detection
    from renom_img.api.utility.target import DataBuilderDetection

    root = "../../../../dataset/train_set/label"
    annotation_list = [os.path.join(root, path) for path in sorted(os.listdir(root))]

    root = "../../../../dataset/train_set/img"
    img_list = [os.path.join(root, path) for path in sorted(os.listdir(root))]
    dist = ImageDistributor(img_list, parse_xml_detection(annotation_list))

    start_t = time.time()
    for x, y in dist.batch(32, DataBuilderDetection((224, 224))):
        print(time.time() - start_t)
        start_t = time.time()
        print(len(x))
