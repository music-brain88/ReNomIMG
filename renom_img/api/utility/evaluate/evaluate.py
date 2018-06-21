import numpy as np
import 


class EvaluatorBase(object):

    def __init__(self, prediction, target):
        self.prediction = prediction
        self.target = target

    def report(self):
        raise NotImplemented


class EvaluatorDetection(EvaluatorBase):

    def __init__(self):
        pass

    def mAP(self):
        raise NotImplemented

    def iou(self):
        raise NotImplemented

    def plot_pr_curve(self):

class EvaluatorClassification(EvaluatorBase):

    def __init__(self):
        pass

    def confusion_matrix(self):
        raise NotImplemented

class EvaluatorSegmentation(EvaluatorBase):

    def __init__(self):
        pass

def get_prec_and_rec():
    pass
