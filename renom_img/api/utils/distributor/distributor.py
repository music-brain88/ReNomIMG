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
        for n in range(len(filelist)//b):
            for i in range(self._num_threads - que.qsize()):
                result = []
                th = LoadThread(filelist[(n+i)*b:(n+i+1)*b], result, self._img_size)
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

    def batch(self, batch_size, callback=lambda x:x):
        ind = 0
        result = []
        total_batch_num = len(self._img_path_list)//batch_size
        event = threading.Event()
        th = ThreadRunner(self._img_path_list, batch_size, result, event, self._img_size, self._num_threads)
        th.start()
        label = None
        while ind < total_batch_num:
            if self._label_list is not None and label is None:
                label = callback(self._label_list[ind*batch_size:(ind+1)*batch_size])
            if len(result[ind]) != batch_size:
                event.clear()
                event.wait()
            else:
                if label is None:
                    yield np.vstack(result[ind])
                else:
                    yield np.vstack(result[ind]), label
                label = None
                ind += 1
        th.join()

class ImageDistributor(ImageDistributorBase):

    def __init__(self, img_path_list, label_list=None, img_size=(224, 224), num_threads=8):
        super(ImageDistributor, self).__init__(img_path_list, label_list, img_size, num_threads)

    def batch(self, batch_size=64): 
        return super(ImageDistributor, self).batch(batch_size)

