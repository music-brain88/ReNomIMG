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
            img = np.asarray(img.resize(self._img_size)).transpose(2, 0, 1)[None, ...]
            self._results.append(img.astype(np.float32))


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
        for n in range(len(filelist) // b):
            for i in range(self._num_threads - que.qsize()):
                result = []
                th = LoadThread(filelist[(n + i) * b:(n + i + 1) * b], result, self._img_size)
                self._results.append(result)
                th.start()
                que.put(th)
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
        ind = 0
        result = []
        total_batch_num = len(self._img_path_list) // batch_size
        event = threading.Event()
        th = ThreadRunner(self._img_path_list, batch_size, result,
                          event, self._img_size, self._num_threads)
        th.start()
        label = None
        while ind < total_batch_num:
            if self._label_list is not None and label is None:
                label = callback(self._label_list[ind * batch_size:(ind + 1) * batch_size])
            if len(result[ind]) != batch_size:
                event.clear()
                event.wait()
            else:
                # Perform argumentation here.
                if label is None:
                    yield np.vstack(result[ind])
                else:
                    yield np.vstack(result[ind]), label
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
        super(ImageDetectionDistributor, self).__init__(img_path_list, label_list, img_size, num_threads)
        self._augumentation = augumentation

    def batch(self, batch_size=64):
        """This returns generator object.
        Make the class label onehot.
        """
        def make_onehot(target):
            N, D = target.shape
            D = D//5
            new_target = np.zeros((N, D*(4+self.class_num)))
            new_target[:, 0::4+self.class_num] = target[:, 0::5]
            new_target[:, 1::4+self.class_num] = target[:, 1::5]
            new_target[:, 2::4+self.class_num] = target[:, 2::5]
            new_target[:, 3::4+self.class_num] = target[:, 3::5]
            active = ((self._class_num+4)*np.arange(D) + target[:, 4::5]).astype(np.int)
            new_target[:, active] = 1
            return new_target 
        return super(ImageDetectionDistributor, self).batch(batch_size)

