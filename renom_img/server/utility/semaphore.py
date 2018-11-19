import time
from threading import Event, Semaphore


class EventSemaphore(object):

    def __init__(self, n):
        self.n = n
        self.max = n
        self.s = Semaphore()

    def acquire(self, event):
        while self.n <= 0 and not event.is_set():
            time.sleep(0.5)

        with self.s:
            self.n -= 1

    def release(self):
        with self.s:
            self.n += 1
        self.n = min(self.n, self.max)
