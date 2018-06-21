import numpy as np
from renom_img.api.utility.evaluate.detection import get_prec_and_rec, get_ap_and_map, get_mean_iou
from renom_img.api.utility.evaluate.classification import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt


class EvaluatorBase(object):

    def __init__(self, prediction, target):
        self.prediction = prediction
        self.target = target

    def report(self, class_names, headers, rows, last_line_heading, last_row=None, digits=3):
        last_line_heading = 'mAP / mean IoU'
        name_width = max(len(cn) for cn in class_names)
        width = max(name_width, len(last_line_heading), digits)

        head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
        report = head_fmt.format('', *headers, width=width)
        report += ' \n\n'

        row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * len(headers) + ' \n'
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += '\n'
        report += row_fmt.format(last_line_heading,
                             *last_row,
                             width=width, digits=digits)
        return report

    def plot_graph(self, x, y, title=None, x_label=None, y_label=None):
        plt.figure()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot(x, y)
        plt.show()

class EvaluatorDetection(EvaluatorBase):
    """
    Evaluator for object detection tasks

    Args:
      gt_list (list):
      pred_list (list): A list of predicted bounding boxes.

    predict_list:
        [
            [ # Objects of 1st image.
                {'box': [x(float), y, w, h], 'class': class_id(int), 'confidence': score},
                {'box': [x(float), y, w, h], 'class': class_id(int), 'confidence': score},
                ...
            ],
            [ # Objects of 2nd image.
                {'box': [x(float), y, w, h], 'class': class_id(int), 'confidence': score},
                {'box': [x(float), y, w, h], 'clas': class_id(int), 'confidence': score},
                ...
            ]
        ]

    Example:
        >>> evaluator = EvaluatorDetection(pred, gt)
        >>> evaluator.mAP()
        >>> evaluator.mean_iou()
    """

    def __init__(self, prediction, target):
        super(EvaluatorDetection, self).__init__(prediction, target)

    def mAP(self, n_class=None, iou_thresh=0.5, n_round_off=3):
        prec, rec = get_prec_and_rec(self.prediction, self.target, n_class, iou_thresh)
        _, mAP = get_ap_and_map(prec, rec, n_round_off)
        return mAP

    def AP(self, n_class=None, iou_thresh=0.5, n_round_off=3):
        prec, rec = get_prec_and_rec(self.prediction, self.target, n_class, iou_thresh)
        AP, _ = get_ap_and_map(prec, rec, n_round_off)
        return AP

    def mean_iou(self, n_class=None, iou_thresh=0.5, n_round_off=3):
        _, mean_iou = get_mean_iou(self.prediction, self.target, n_class, iou_thresh, n_round_off)
        return mean_iou

    def iou(self, n_class=None, iou_thresh=0.5, n_round_off=3):
        iou, _ = get_mean_iou(self.prediction, self.target, n_class, iou_thresh, n_round_off)
        return iou

    def plot_pr_curve(self, n_class=None, iou_thresh=0.5, class_names=None):
        """
        Args:
            class_names: key in a precision(recall) list
                Plotting precision-recall curve of specified class name
        """
        prec, rec = get_prec_and_rec(self.prediction, self.target, n_class, iou_thresh)
        if not isinstance(class_names, list) and class_names is not None:
            class_names = [class_names]

        if class_names:
            for c in class_names:
                p = prec[c]
                r = rec[c]
                if r is None or len(r) == 0:
                    continue
                self.plot_graph(r, p, c, 'Recall', 'Precision')
        else:
            for c in list(prec.keys()):
                p = prec[c]
                r = rec[c]
                if r is None or len(r) == 0:
                    continue
                self.plot_graph(r, p, c, 'Recall', 'Precision')

    def detection_report(self, digits=3):
        AP = self.AP()
        class_names = list(AP.keys())
        AP = list(AP.values())
        mAP = self.mAP()
        iou = list(self.iou().values())
        mean_iou = self.mean_iou()

        headers = ["AP", "IoU"]
        rows = zip(class_names, AP, iou)
        last_line_heading = 'mAP / mean IoU'
        last_row = (mAP, mean_iou)

        return self.report(class_names, headers, rows, last_line_heading, last_row, digits)

class EvaluatorClassification(EvaluatorBase):

    def __init__(self, prediction, target):
        super(EvaluatorClassification, self).__init__(prediction, target)

    def precision(self):
        precision = precision_score(self.prediction, self.target)
        return precision

    def recall(self):
        recall = recall_score(self.prediction, self.target)
        return recall

    def accuracy(self):
        accuracy = accuracy_score(self.prediction, self.target)
        return accuracy

    def confusion_matrix(self):
        raise NotImplemented

class EvaluatorSegmentation(EvaluatorBase):
    def __init__(self):
        pass
