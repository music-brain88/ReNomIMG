import asyncio
import requests
import urllib.request

import numpy as np
from renom_img.server.model_wrapper.yolo import WrapperYoloDarknet
from renom_img.api.utility.misc.download import download

pre = 0


class Detector(object):

    def __init__(self, project_id=1, url="http://localhost", port='8080'):
        self._pj_id = project_id
        self._url = url
        self._port = port
        self._model = None
        self._model_info = {}

    def pull(self, ):
        url = self._url + ':' + self._port
        download_weight_api = "/api/renom_img/v1/projects/{}/deployed_model".format(self._pj_id)
        download_param_api = "/api/renom_img/v1/projects/{}/deployed_model_info".format(self._pj_id)
        download_weight_api = url + download_weight_api
        download_param_api = url + download_param_api
        download(download_weight_api, "deployed_model.h5")

        ret = requests.get(download_param_api).json()

        # TODO: Check algorithm.
        cell = ret["algorithm_params"]["cells"]
        bbox = ret["algorithm_params"]["bounding_box"]
        img_w = ret["hyper_parameters"]["image_width"]
        img_h = ret["hyper_parameters"]["image_height"]
        self._model = WrapperYoloDarknet(0, cell, bbox, (img_h, img_w))
        self._model.load("deployed_model.h5")

        self._model_info = {
            "Image size": "{}x{}".format(img_w, img_h),
            "Num class": "{}".format(self._model._num_class)
        }

    def predict(self, x, normalize=True):
        assert self._model
        if normalize:
            x = x / 255. - 0.5
        return self._model.get_bbox(self._model(self._model.freezed_forward(x)).as_ndarray())

    @property
    def model_info(self):
        return self._model_info
