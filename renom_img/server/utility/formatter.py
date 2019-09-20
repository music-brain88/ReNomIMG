from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass

import pandas as pd
from pandas.io.json import json_normalize

from renom_img.server import Task


def get_formatter_resolver(task_id):
    if task_id == Task.CLASSIFICATION.value:
        resolver = FormatterResolver([ClassificationCsvFormatter()])
    elif task_id == Task.DETECTION.value:
        resolver = FormatterResolver([DetectionCsvFormatter()])
    elif task_id == Task.SEGMENTATION.value:
        resolver = FormatterResolver([SegmentationCsvFormatter()])
    return resolver


class FormatterResolver(object):
    def __init__(self, formatters):
        self.formatters = formatters

    def resolve(self, format):
        for formatter in self.formatters:
            if formatter.format == format:
                return formatter


class FormatterBase(with_metaclass(ABCMeta, object)):
    def __init__(self):
        pass

    @abstractmethod
    def to_df(self, data):
        pass


class DetectionCsvFormatter(FormatterBase):
    def __init__(self):
        self.format = "csv"

    def to_df(self, data):
        img_path = data["img"]
        sizes = data["size"]
        prediction = data["prediction"]

        ret = []
        for img, size, pred in zip(img_path, sizes, prediction):
            ret.append({
                'path': img,
                'size': size,
                'predictions': pred
            })
        df = pd.DataFrame.from_dict(json_normalize(ret), orient='columns')
        return df


class ClassificationCsvFormatter(FormatterBase):
    def __init__(self):
        self.format = "csv"

    def to_df(self, data):
        img_path = data["img"]
        sizes = data["size"]
        prediction = data["prediction"]
        scores = data["scores"]

        ret = []
        for img, size, pred, score in zip(img_path, sizes, prediction, scores):
            ret.append({
                'path': img,
                'size': size,
                'predictions': pred["class"],
                'prediction scores': score
            })

        df = pd.DataFrame.from_dict(json_normalize(ret), orient='columns')
        df = df[['path','predictions','size','prediction scores']]
        return df


class SegmentationCsvFormatter(FormatterBase):
    def __init__(self):
        self.format = "csv"

    def to_df(self, data):
        img_path = data["img"]
        sizes = data["size"]
        prediction = data["prediction"]

        ret = []
        for img, size, pred in zip(img_path, sizes, prediction):
            ret.append({
                'path': img,
                'size': size,
                'predictions': pred
            })

        df = pd.DataFrame.from_dict(json_normalize(ret), orient='columns')
        return df
