import os
import threading
import numpy as np
from PIL import Image
from queue import Queue


class LoadThread(threading.Thread):

    def __init__(self, filenames, results, img_size=(224, 224), color="RGB"):
        super(LoadThread, self).__init__()
        self._filenames = filenames
        self._results = results
        self._img_size = img_size

    def run(self):
        for filenames in self._filenames:
            img = Image.open(filenames)
            img.load()
            self._results.append(img)


class ThreadRunner(threading.Thread):

    def __init__(self, filelist, batch_size, results, event, img_size, num_threads=4):
        super(ThreadRunner, self).__init__()
        self._event = event
        self._results = results
        self._img_size = img_size
        self._filelist = filelist
        self._batch_size = batch_size
        self._num_threads = num_threads

    def run(self):
        que = Queue()
        b = self._batch_size
        filelist = self._filelist
        b_count = 0
        for n in range(len(filelist) // b):
            for i in range(self._num_threads - que.qsize()):
                result = []
                th = LoadThread(filelist[b_count * b:(b_count + 1) * b], result, self._img_size)
                self._results.append(result)
                th.start()
                que.put(th)
                b_count += 1
            que.get().join()
            self._event.set()

        for _ in range(que.qsize()):
            que.get().join()
            self._event.set()


class ImageDistributorBase(object):

    def __init__(self, img_path_list, label_list=None, img_size=(224, 224), num_threads=8):
        self._img_path_list = img_path_list
        self._label_list = label_list
        self._num_threads = num_threads
        self._img_size = img_size
        self._class_num = int(np.max([d[::5] for d in label_list]))

    def __len__(self):
        return len(self._img_path_list)

    @property
    def img_path_list(self):
        return self._img_path_list

    @property
    def class_num(self):
        return self._class_num

    def batch(self, batch_size, callback=lambda x: x):
        N = len(self)
        ind = 0
        result = []
        size_w, size_h = self._img_size
        total_batch_num = len(self._img_path_list) // batch_size
        event = threading.Event()
        perm = np.random.permutation(N)
        label_list = self._label_list[perm]
        img_list = np.array(self._img_path_list)[perm]
        th = ThreadRunner(img_list, batch_size, result,
                          event, self._img_size, self._num_threads)
        th.start()
        label = None
        while ind < total_batch_num:
            if self._label_list is not None and label is None:
                label = callback(label_list[ind * batch_size:(ind + 1) * batch_size])
            if len(result[ind]) != batch_size:
                event.clear()
                event.wait()
            else:
                # Perform argumentation here.
                # Resize.
                X = np.vstack([np.asarray(img.resize((size_w, size_h))).transpose(
                    2, 0, 1).astype(np.float32)[np.newaxis].copy() for img in result[ind]])
                if label is None:
                    yield X
                else:
                    sizes = np.array([(img.size[0] / size_w, img.size[1] / size_h)
                                      for img in result[ind]])
                    resized_label = np.zeros((batch_size, len(label[0])))
                    resized_label[:, 0::5] = label[:, 0::5] / sizes[:, 0][..., None]
                    resized_label[:, 1::5] = label[:, 1::5] / sizes[:, 1][..., None]
                    resized_label[:, 2::5] = label[:, 2::5] / sizes[:, 0][..., None]
                    resized_label[:, 3::5] = label[:, 3::5] / sizes[:, 1][..., None]
                    resized_label[:, 4::5] = label[:, 4::5]
                    yield X, resized_label
                label = None
                ind += 1
        th.join()


class ImageDistributor(ImageDistributorBase):

    def __init__(self, img_path_list, label_list=None, img_size=(224, 224), augumentation=None, num_threads=8):
        super(ImageDistributor, self).__init__(img_path_list, label_list, img_size, num_threads)
        self._augumentation = augumentation

    def batch(self, batch_size=64):
        """This returns generator object.
        """
        return super(ImageDistributor, self).batch(batch_size)


class ImageDetectionDistributor(ImageDistributorBase):

    def __init__(self, img_path_list, label_list=None, img_size=(224, 224), augumentation=None, num_threads=8):
        super(ImageDetectionDistributor, self).__init__(
            img_path_list, label_list, img_size, num_threads)
        self._augumentation = augumentation

    def batch(self, batch_size=64):
        """This returns generator object.
        Make the class label onehot.
        """
        return super(ImageDetectionDistributor, self).batch(batch_size)
