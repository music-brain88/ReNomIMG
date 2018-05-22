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
        for n in range(int(np.ceil(len(filelist)/b))):
            for i in range(self._num_threads - que.qsize()):
                result = []
                th = LoadThread(filelist[b_count * b:(b_count + 1) * b], result, self._img_size)
                self._results.append(result)
                th.start()
                que.put(th)
                b_count += 1
            que.get().join()
            self._event.set()
            if b_count >= int(np.ceil(len(filelist)/b)):
                break

        for _ in range(que.qsize()):
            que.get().join()
            self._event.set()


class ImageDistributorBase(object):

    def __init__(self, img_path_list, label_list=None, img_size=(224, 224), augmentation=None, num_threads=8):
        self._img_path_list = img_path_list
        self._label_list = label_list
        self._num_threads = num_threads
        self._img_size = img_size
        self._class_num = None
        self._augmentation = augmentation

    def __len__(self):
        return len(self._img_path_list)

    @property
    def img_path_list(self):
        return self._img_path_list

    @property
    def class_num(self):
        return self._class_num

    def batch(self, batch_size, shuffle=True):
        N = len(self)
        ind = 0
        result = []
        size_w, size_h = self._img_size
        total_batch_num = int(np.ceil(len(self._img_path_list)/batch_size))
        event = threading.Event()
        if shuffle:
            if N < 100000:
                perm = np.random.permutation(N)
            else:
                perm = np.random.randint(0, N, size=(N, )) 
            label_list = self._label_list[perm]
            img_list = np.array(self._img_path_list)[perm]
        else:
            label_list = self._label_list
            img_list = np.array(self._img_path_list)

        each_batch_size = [batch_size if N-i*batch_size > batch_size else N-i*batch_size \
             for i in range(total_batch_num)]
        th = ThreadRunner(img_list, batch_size, result,
                          event, self._img_size, self._num_threads)
        th.start()
        label = None
        while ind < total_batch_num:
            if self._label_list is not None and label is None:
                label = callback(label_list[ind * batch_size:(ind + 1) * batch_size])

            if len(result[ind]) != each_batch_size[ind]:
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
                    yield X, label
                label = None
                ind += 1
        th.join()


class ImageDistributor(ImageDistributorBase):

    def __init__(self, img_path_list, label_list=None, img_size=(224, 224), augmentation=None, num_threads=8):
        super(ImageDistributor, self).__init__(img_path_list, label_list, img_size, augmentation, num_threads)


class ImageDetectionDistributor(ImageDistributorBase):

    def __init__(self, img_path_list, label_list=None, img_size=(224, 224), augmentation=None, num_threads=8):
        super(ImageDetectionDistributor, self).__init__(
            img_path_list, label_list, img_size, augmentation, num_threads)
        if label_list is not None:
            self._class_num = int(np.max([d[::5] for d in label_list]))

    def batch(self, batch_size=64, shuffle=True):
        """This returns generator object.
        Make the class label onehot.
        """
        N = len(self)
        ind = 0
        result = []
        size_w, size_h = self._img_size
        total_batch_num = int(np.ceil(len(self._img_path_list)/batch_size))
        event = threading.Event()
        if shuffle:
            if N < 100000:
                perm = np.random.permutation(N)
            else:
                perm = np.random.randint(0, N, size=(N, ))
            label_list = self._label_list[perm]
            img_list = np.array(self._img_path_list)[perm]
        else:
            label_list = self._label_list
            img_list = np.array(self._img_path_list)
        each_batch_size = [batch_size if N-i*batch_size > batch_size else N-i*batch_size \
             for i in range(total_batch_num)]
        th = ThreadRunner(img_list, batch_size, result,
                          event, self._img_size, self._num_threads)
        th.start()
        label = None
        while ind < total_batch_num:
            if self._label_list is not None and label is None:
                label = label_list[ind * batch_size:(ind + 1) * batch_size]
            if len(result[ind]) != each_batch_size[ind]:
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
                    resized_label = np.zeros((len(label), len(label[0])))
                    resized_label[:, 0::5] = label[:, 0::5] / sizes[:, 0][..., None]
                    resized_label[:, 1::5] = label[:, 1::5] / sizes[:, 1][..., None]
                    resized_label[:, 2::5] = label[:, 2::5] / sizes[:, 0][..., None]
                    resized_label[:, 3::5] = label[:, 3::5] / sizes[:, 1][..., None]
                    resized_label[:, 4::5] = label[:, 4::5]
                    if self._augmentation is None:
                        yield X, resized_label
                    else:
                        yield self._augmentation(X, resized_label, mode="detection")
                label = None
                ind += 1
        th.join()
