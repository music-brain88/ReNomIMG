import numpy as np
from .detection import get_prec_and_rec, get_ap_and_map, get_mean_iou
from .classification import precision_score, recall_score, f1_score, accuracy_score
from .segmentation import segmentation_iou, segmentation_precision, segmentation_recall, segmentation_f1, get_segmentation_metrics
from collections import defaultdict


class EvaluatorBase(object):

    def __init__(self, prediction, target):
        self.prediction = prediction
        self.target = target

    def build_report(self, class_names, headers, rows, last_line_heading, row_fmt, last_row=None, digits=3):
        name_width = max(len(str(cn)) for cn in class_names)
        width = max(name_width, len(last_line_heading), digits * 2)

        head_fmt = '{:>{width}s} ' + ' {:>12}' * (len(headers) - 1) + '{:>16}'
        report = head_fmt.format('', *headers, width=width)
        report += ' \n\n'

        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += '\n'
        report += row_fmt.format(last_line_heading,
                                 *last_row,
                                 width=width, digits=digits)
        return report

    def plot_graph(self, x, y, title=None, x_label=None, y_label=None):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot(x, y)
        plt.show()


class EvaluatorDetection(EvaluatorBase):
    """ Evaluator for object detection tasks

    Args:
        gt_list (list): A list of ground truth.
        pred_list (list): A list of prediction. The format is as follows

        predict_list:
            [
                [ # Objects of 1st image.
                    {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score},
                    {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score},
                    ...
                ],
                [ # Objects of 2nd image.
                    {'box': [x(float), y, w, h], 'class': class_id(int), 'score': score},
                    {'box': [x(float), y, w, h], 'clas': class_id(int), 'score': score},
                    ...
                ]
            ]

    Example:
            >>> evaluator = EvaluatorDetection(pred, gt)
            >>> evaluator.mAP()
            >>> evaluator.mean_iou()
    """

    def __init__(self, prediction, target, n_class=None):
        super(EvaluatorDetection, self).__init__(prediction, target)
        self.n_class = n_class

    def mAP(self, iou_thresh=0.5, digits=3):
        """ mAP (mean Average Precision)
        Args:
            iou_thresh: IoU threshold. The default value is 0.5.
            digits: The number of decimal.

        Returns:
            mAP (float)
        """

        prec, rec, _, _ = get_prec_and_rec(self.prediction, self.target, self.n_class, iou_thresh)
        _, mAP = get_ap_and_map(prec, rec, digits)
        return mAP

    def AP(self, iou_thresh=0.5, digits=3):
        """ AP (Average Precision for each class)
        Args:
            iou_thresh: IoU threshold. The default value is 0.5.
            digits: The number of decimal.

        Returns:
            {
                class_name1(str): AP1 (float),
                class_name2(str): AP2 (float),
                class_name3(str): AP3 (float),
            }
        """

        prec, rec, _, _ = get_prec_and_rec(self.prediction, self.target, self.n_class, iou_thresh)
        AP, _ = get_ap_and_map(prec, rec, digits)
        return AP

    def mean_iou(self, iou_thresh=0.5, digits=3):
        """ mean IoU for all classes
        Args:
            iou_thresh: IoU threshold. The default value is 0.5.
            digits: The number of decimal.

        returns:
            mean_iou (float)
        """
        _, mean_iou = get_mean_iou(self.prediction, self.target, self.n_class, iou_thresh, digits)
        return mean_iou

    def iou(self, iou_thresh=0.5, digits=3):
        """ IoU for each class
        Args:
            iou_thresh: IoU threshold. The default value is 0.5.
            digits: The number of decimal.

        returns:
            {
                class_name1(str): iou1 (float),
                class_name2(str): iou2 (float),
                class_name3(str): iou3 (float),
            }
        """

        iou, _ = get_mean_iou(self.prediction, self.target, self.n_class, iou_thresh, digits)
        return iou

    def plot_pr_curve(self, iou_thresh=0.5, class_names=None):
        """ Plot a precision-recall curve.
        Args:
            iou_thresh: IoU threshold. The default value is 0.5.
            class_names: List of keys in a prediction list or string if you output precision-recall curve of only one class. This specifies which precision-recall curve of classes to output.
        """

        prec, rec, _, _ = get_prec_and_rec(self.prediction, self.target, self.n_class, iou_thresh)
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

    def prec_rec(self, iou_thresh=0.5):
        """ Return precision and recall for each class
        Args:
            iou_thresh: IoU threshold. The default value is 0.5.
            digits: The number of decimal of output values

        Returns:
            precision(dictionary): {class_id1(int): [0.5, 0.3,....], class_id2(int): [0.9....]}
            recall(dictionary): {class_id1(int): [0.5, 0.3,....], class_id2(int): [0.9....]}
        """

        prec, rec, _, _ = get_prec_and_rec(self.prediction, self.target, self.n_class, iou_thresh)
        return prec. rec

    def report(self, iou_thresh=0.5, digits=3):
        """ Output a table whcih shows AP, IoU, the number of predicted instances for each class, and the number of ground truth instances for each class.
        Args:
            iou_thresh: IoU threshold. The default value is 0.5.
            class_names: List of keys in a prediction list or string if you output precision-recall curve of only one class. This specifies which precision-recall curve of classes to output.

        Returns:
                                AP         IoU        #pred/#target
            class_name1:      0.091      0.561            1/13
            class_name2:      0.369      0.824            6/15
                ....
            mAP / mean IoU    0.317      0.698          266/686
        """

        prec, rec, n_pred, n_pos_list = get_prec_and_rec(
            self.prediction, self.target, self.n_class, iou_thresh)
        AP, mAP = get_ap_and_map(prec, rec, digits)
        iou = self.iou()
        class_names = list(AP.keys())
        mean_iou = self.mean_iou()

        headers = ["AP", "IoU", "  #pred/#target"]
        rows = []
        for c in class_names:
            rows.append((str(c), AP[c], iou[c], n_pred[c], n_pos_list[c]))
        last_line_heading = 'mAP / mean IoU'
        last_row = (mAP, mean_iou, np.sum(list(n_pred.values())), np.sum(list(n_pos_list.values())))
        row_fmt = '{:>{width}s} ' + ' {:>12.{digits}f}' * \
            (len(headers) - 1) + ' {:>12d}/{:d}' + ' \n'

        return self.build_report(class_names, headers, rows, last_line_heading, row_fmt, last_row, digits)


class EvaluatorClassification(EvaluatorBase):

    def __init__(self, prediction, target):
        super(EvaluatorClassification, self).__init__(prediction, target)

    def precision(self):
        """
        Returns:
            precision(float)
        """
        precision, mean_precision = precision_score(self.prediction, self.target)
        return precision, mean_precision

    def recall(self):
        """
        Returns:
            recall(float)
        """
        recall, mean_recall = recall_score(self.prediction, self.target)
        return recall, mean_recall

    def accuracy(self):
        accuracy = accuracy_score(self.prediction, self.target)
        return accuracy

    def f1(self):
        """
        Returns:
            f1_score(dict):
            mean_f1_score(float):
        """
        f1, mean_f1 = f1_score(self.prediction, self.target)
        return f1, mean_f1

    def report(self, digits=3):
        precision, mean_precision = self.precision()
        recall, mean_recall = self.recall()
        f1, mean_f1 = self.f1()
        accuracy = self.accuracy()
        class_names = list(precision.keys())

        tp = defaultdict(int)
        true_sum = defaultdict(int)

        for p, t in zip(self.prediction, self.target):
            true_sum[t] += 1
            if p == t:
                tp[p] += 1

        headers = ["Precision", "Recall", "F1 score", "#pred/#target"]
        rows = []
        for c in class_names:
            rows.append((str(c), precision[c], recall[c], f1[c], tp[c], true_sum[c]))
        last_line_heading = 'Average'
        last_row = (mean_precision, mean_recall, mean_f1, np.sum(
            list(tp.values())), np.sum(list(true_sum.values())))
        row_fmt = '{:>{width}s} ' + ' {:>12.{digits}f}' * \
            (len(headers) - 1) + ' {:>12d}/{:d}' + ' \n'

        report = self.build_report(class_names, headers, rows,
                                   last_line_heading, row_fmt, last_row, digits)
        report += '\n'
        report += ('Accuracy' + ' {:>12.{digits}f}'.format(accuracy, digits=digits))
        return report


class EvaluatorSegmentation(EvaluatorBase):
    def __init__(self, prediction, target):
        super(EvaluatorSegmentation, self).__init__(prediction, target)

    def iou(self, background_class=0):
        """ Returns iou for each class
        """

        iou, mean_iou = segmentation_iou(
            self.prediction, self.target, background_class=background_class)
        return iou, mean_iou

    def precision(self, background_class=0, digits=3):
        """ Returns precision for each class
        """
        precision, mean_precision = segmentation_precision(self.prediction,
                                                           self.target,
                                                           digits=digits,
                                                           background_class=background_class)
        return precision, mean_precision

    def recall(self, background_class=0, digits=3):
        """ Returns recall for each class and mean recall
        """
        recall, mean_recall = segmentation_recall(self.prediction,
                                                  self.target,
                                                  digits=digits,
                                                  background_class=background_class)

    def f1(self, background_class=0, digits=3):
        """ Returns f1 for each class and mean f1 score
        """
        f1, mean_f1 = segmentation_f1(self.prediction,
                                      self.target,
                                      digits=digits,
                                      background_class=background_class)

    def report(self, background_class=0, digits=3):
        precision, mean_precision, \
                recall, mean_recall, \
                f1, mean_f1, iou, \
                mean_iou, tp, true_sum = get_segmentation_metrics(self.prediction,
                                                                  self.target,
                                                                  digits=digits,
                                                                  background_class=background_class)

        headers = ["IoU", "Precision", "Recall", "F1 score", "#pred/#target"]
        rows = []
        class_names = list(precision.keys())

        for c in class_names:
            rows.append((str(c), iou[c], precision[c], recall[c], f1[c], tp[c], true_sum[c]))
        last_line_heading = 'Average'
        last_row = (mean_iou, mean_precision, mean_recall, mean_f1, np.sum(
            list(tp.values())), np.sum(list(true_sum.values())))
        row_fmt = '{:>{width}s} ' + ' {:>12.{digits}f}' * \
            (len(headers) - 1) + ' {:>12d}/{:d}' + ' \n'

        report = self.build_report(class_names, headers, rows,
                                   last_line_heading, row_fmt, last_row, digits)
        report += '\n'
        return report


