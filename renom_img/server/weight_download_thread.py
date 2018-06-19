#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import urllib.request


class WeightDownloadThread(threading.Thread):

    def __init__(self, thread_id, url, file_name):
        super(WeightDownloadThread, self).__init__()
        self.setDaemon(False)
        self.thread_id = thread_id
        self.url = url
        self.file_name = file_name
        self.percentage = 0
        self.i = 1
        self.step = 10

    def progress(self, block_count, block_size, total_size):
        self.percentage = 100.0 * block_count * block_size / total_size

    def run(self):
        urllib.request.urlretrieve(self.url, self.file_name, self.progress)
