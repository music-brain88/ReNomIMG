import renom as rm
import numpy as np


class BaseOptimizer(object):

    def __init__(self, total_batch_iteration=None, total_epoch_iteration=None):
        self.opt = rm.Sgd()
        if not (None in [total_epoch_iteration, total_batch_iteration]):
            self.setup(total_batch_iteration, total_epoch_iteration)

    def __call__(self, dy, node):
        return self.opt(dy, node)

    def setup(self, total_batch_iteration, total_epoch_iteration):
        self.opt = rm.Sgd()
        self.total_batch_iteration = total_batch_iteration
        self.total_epoch_iteration = total_epoch_iteration
        self.total_iteration = total_batch_iteration * total_epoch_iteration
        self.nth_batch_iteration = 0
        self.nth_epoch_iteration = 0

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        assert self.total_epoch_iteration >= nth_epoch, \
            "The max epoch iteration count is {} but set {}".format(
                self.total_epoch_iteration, nth_epoch)
        assert self.total_batch_iteration >= nth_batch, \
            "The max batch iteration count is {} but set {}".format(
                self.total_batch_iteration, nth_batch)
        self.nth_batch_iteration = nth_batch
        self.nth_epoch_iteration = nth_epoch


class OptimizerYolov1(BaseOptimizer):

    def __init__(self, total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerYolov1, self).__init__(total_batch_iteration, total_epoch_iteration)
        self.opt = rm.Sgd(0.0005, 0.9)

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerYolov1, self).setup(total_batch_iteration, total_epoch_iteration)
        sch1 = int(self.total_iteration * 0.005)
        sch2 = int(self.total_iteration * 0.012)
        sch3 = int(self.total_iteration * 0.017)
        sch4 = int(self.total_iteration * 0.055)
        sch5 = self.total_iteration - (sch1 + sch2 + sch3 + sch4)
        self.schedule = [0.0005] * sch1 + [0.00125] * sch2 + \
            [0.0025] * sch3 + [0.005] * sch4 + [0.005] * sch5

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerYolov1, self).set_information(
            nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list)
        self.opt._lr = self.schedule[int(nth_epoch * nth_batch)]


class OptimizerCyclicSgd(BaseOptimizer):
    pass


class OptimizerScheduled(BaseOptimizer):
    pass
