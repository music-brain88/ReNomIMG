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


class OptimizerDeeplab(BaseOptimizer):
    def __init__(self, lr_initial=1e-4, lr_power=0.9, total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerDeeplab, self).__init__(total_batch_iteration, total_epoch_iteration)
        self.lr_initial = lr_initial
        self.lr_power = lr_power
        self.current_iterations = 0

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerDeeplab, self).setup(total_batch_iteration, total_epoch_iteration)
        self.total_iterations = total_batch_iteration * total_epoch_iteration
        self.opt._lr = self.lr_initial*((1 - self.current_iterations/self.total_iterations)**self.lr_power)

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerDeeplab, self).set_information(nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list)
        self.current_iterations = self.total_batch_iteration*nth_epoch + nth_batch
        self.opt._lr = self.lr_initial*((1 - self.current_iterations/self.total_iterations)**self.lr_power)

class OptimizerUNet(BaseOptimizer):
    def __init__(self,total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerUNet, self).__init__(total_batch_iteration, total_epoch_iteration)

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerUNet, self).setup(total_batch_iteration, total_epoch_iteration)
        self.opt._lr = 1e-2
        self.opt._momentum = 0.9

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerUNet, self).set_information(nth_batch, nth_epoch, avg_train_loss_list,avg_valid_loss_list)

        ind1 = int(self.total_epoch_iteration * 0.5)
        ind2 = int(self.total_epoch_iteration * 0.3) + ind1 + 1 
        if nth_epoch == ind1:
            self.opt._lr = 6e-3
        elif nth_epoch == ind2:
            self.opt._lr = 1e-3
        else:
            self.opt._lr = 1e-2

class OptimizerTernausNet(BaseOptimizer):
    def __init__(self,total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerTernausNet, self).__init__(total_batch_iteration, total_epoch_iteration)

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerTernausNet, self).setup(total_batch_iteration, total_epoch_iteration)
        self.opt = rm.Adam(1e-4)

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerTernausNet, self).set_information(nth_batch, nth_epoch, avg_train_loss_list,avg_valid_loss_list)
        pass

class FCN_Optimizer(BaseOptimizer):
    def __init__(self,total_batch_iteration=None, total_epoch_iteration=None):
        super(FCN_Optimizer, self).__init__(total_batch_iteration, total_epoch_iteration)

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(FCN_Optimizer, self).setup(total_batch_iteration, total_epoch_iteration)
        self.opt._lr = 1e-5
        self.opt._momentum = 0.9

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(FCN_Optimizer, self).set_information(nth_batch, nth_epoch, avg_train_loss_list,avg_valid_loss_list)
        pass

class OptimizerDarknet19(BaseOptimizer):
    def __init__(self,total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerDarknet19, self).__init__(total_batch_iteration, total_epoch_iteration)

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerDarknet19, self).setup(total_batch_iteration, total_epoch_iteration)
        self.opt._lr = 0.01
        self.opt._momentum = 0.9

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerDarknet19, self).set_information(nth_batch, nth_epoch, avg_train_loss_list,avg_valid_loss_list)

        if nth_epoch == int(0.3 * self.total_epoch_iteration):
            self.opt._lr = 0.001
        elif nth_epoch == int(0.8 * self.total_epoch_iteration):
            self.opt._lr = 0.0001
 

class OptimizerDenseNet(BaseOptimizer):
    def __init__(self,total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerDenseNet, self).__init__(total_batch_iteration, total_epoch_iteration)

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerDenseNet, self).setup(total_batch_iteration, total_epoch_iteration)
        self.opt._lr = 0.1
        self.opt._momentum = 0.9

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerDenseNet, self).set_information(nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list)

        if nth_epoch == 30:
            self.opt._lr = 0.01
        elif nth_epoch == 60:
            self.opt._lr = 0.001


class OptimizerInception(BaseOptimizer):
    def __init__(self, version,total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerInception, self).__init__(total_batch_iteration, total_epoch_iteration)
        self.version = version

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerInception, self).setup(total_batch_iteration, total_epoch_iteration)
        self.opt._lr = 0.045
        self.opt._momentum = 0.9        

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerInception, self).set_information(nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list)

        if self.version == 1 or self.version == 4:
            if nth_epoch % 8 == 0:
                lr = self.opt._lr * 0.94
                self.opt._lr = lr
        elif self.version == 2:
            if nth_epoch % 2 == 0:
                lr = self.opt._lr * 0.94
                self.opt._lr = lr 


class OptimizerVGG(BaseOptimizer):
    def __init__(self, total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerVGG, self).__init__(total_batch_iteration, total_epoch_iteration)

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerVGG, self).setup(total_batch_iteration, total_epoch_iteration)
        self.opt._lr = 0.01
        self.opt._momentum = 0.9

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerVGG, self).set_information(nth_batch, nth_epoch, avg_train_loss_list,
                                                     avg_valid_loss_list)

        if nth_epoch == 0:
            self.opt._lr = 0.00001 + (0.001 - 0.00001) * nth_batch / self.total_batch_iteration



class OptimizerResNeXt(BaseOptimizer):
    def __init__(self, plateau,total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerResNeXt, self).__init__(total_batch_iteration, total_epoch_iteration)
        self.plateau = plateau

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerResNeXt, self).setup(total_batch_iteration, total_epoch_iteration)
        self.patience = 15
        self.min_lr = 1e-6
        self.counter = 0
        self.factor = np.sqrt(0.1)
        self.opt._lr = 0.1
        self.opt._momentum = 0.9

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerResNeXt, self).set_information(nth_batch, nth_epoch, avg_train_loss_list,
                                                     avg_valid_loss_list)

        if self.plateau:
             if len(avg_valid_loss_list) >= 2 and nth_batch == 0:
                if avg_valid_loss_list[-1] > min(avg_valid_loss_list):
                    self.counter +=1
                    new_lr = self.opt._lr * self.factor
                    if self.counter > self.patience and new_lr > self.min_lr:
                        self.opt._lr = new_lr
                        self.counter = 0
                else:
                    self.counter = 0

class OptimizerResNet(BaseOptimizer):

    def __init__(self, plateau, total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerResNet, self).__init__(total_batch_iteration, total_epoch_iteration)
        self.plateau = plateau
    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerResNet, self).setup(total_batch_iteration, total_epoch_iteration)
        self.patience = 15
        self.min_lr = 1e-6
        self.counter = 0
        self.factor = np.sqrt(0.1)
        self.opt._lr = 0.1
        self.opt._momentum = 0.9

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerResNet, self).set_information(nth_batch, nth_epoch, avg_train_loss_list,
                                                    avg_valid_loss_list)
        if self.plateau:
            if len(avg_valid_loss_list) >= 2 and nth_batch == 0:
                if avg_valid_loss_list[-1] > min(avg_valid_loss_list):
                    self.counter +=1
                    new_lr = self.opt._lr * self.factor
                    if self.counter > self.patience and new_lr > self.min_lr:
                        self.opt._lr = new_lr
                        self.counter = 0
                else:
                    self.counter = 0

class OptimizerSSD(BaseOptimizer):
    def __init__(self, total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerSSD, self).__init__(total_batch_iteration, total_epoch_iteration)

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerSSD, self).setup(total_batch_iteration, total_epoch_iteration)
        self.opt._lr = 1e-3
        self.opt._momentum = 0.9

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerSSD, self).set_information(
            nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list)
        
        if nth_epoch <1 :
            self.opt._lr = (1e-3 - 1e-5) / self.total_batch_iteration * nth_batch + 1e-5
        elif nth_epoch < 60 / 160. * self.total_epoch_iteration:
            self.opt._lr = 1e-3
        elif nth_epoch < 100 / 160. * self.total_epoch_iteration:
            self.opt._lr = 1e-4
        else:
            self.opt._lr = 1e-5

#---------------------------------------------------------------------------------------------
class OptimizerYolov2(BaseOptimizer):

    def __init__(self, total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerYolov2, self).__init__(total_batch_iteration, total_epoch_iteration)

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerYolov2, self).setup(total_batch_iteration, total_epoch_iteration)
        self.flag = True
        self.burn_in = 1000
        self.scale=[0.1,0.1]
        self.steps=[40000,60000]
        self.power=4
        self.learning_rate = 0.001
        self.opt._lr = self.learning_rate
        self.opt._momentum = 0.9

    def set_information(self, nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list):
        super(OptimizerYolov2, self).set_information(
            nth_batch, nth_epoch, avg_train_loss_list, avg_valid_loss_list)

        current_batch_num = (self.total_batch_iteration * nth_epoch) + (nth_batch +1)

        if current_batch_num > int(0.3 * self.total_iteration) and hasattr(self,"flag"):
            self.flag = False

        if current_batch_num <= self.burn_in:
            self.opt._lr = self.learning_rate * np.round(np.power(float(current_batch_num/self.burn_in),self.power),3)
            return 
        rate = self.opt._lr
        for i in range(len(self.steps)):
            if self.steps[i] == current_batch_num:
                rate *= self.scale[i]
                self.opt._lr = rate
 
#---------------------------------------------------------------------------------------------


class OptimizerYolov1(BaseOptimizer):

    def __init__(self, total_batch_iteration=None, total_epoch_iteration=None):
        super(OptimizerYolov1, self).__init__(total_batch_iteration, total_epoch_iteration)

    def setup(self, total_batch_iteration, total_epoch_iteration):
        super(OptimizerYolov1, self).setup(total_batch_iteration, total_epoch_iteration)
        self.opt._lr = 0.0005
        self.opt._momentum = 0.9
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
