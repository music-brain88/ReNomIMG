from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass

import pandas as pd
from pandas.io.json import json_normalize

from renom_img.server import Task


def get_formatter_resolver(task_id):
    if task_id == Task.CLASSIFICATION.value:
        pass
    elif task_id == Task.DETECTION.value:
        resolver = FormatterResolver([DetectionCsvFormatter()])
    elif task_id == Task.SEGMENTATION.value:
        pass
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
    def format(self, data):
        pass


class DetectionCsvFormatter(FormatterBase):
    def __init__(self):
        self.format = "csv"

    def format(data):
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
