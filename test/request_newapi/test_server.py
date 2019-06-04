import os
import time
import pickle
import random
import json
import pytest
import threading

from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor


### New API Dummy Data ###
CREATED = "2019-03-13T00:49:05"
CLASS_MAP = ["dog", "cat"]
CLASS_INFO = {
    "class_map": CLASS_MAP,
    "class_ratio": [0.5, 0.5],
    "train_ratio": [0.8, 0.8],
    "valid_ratio": [0.2, 0.2],
    "test_ratio": [0.0, 0.0],
    "train_img_num": 800,
    "valid_img_num": 200,
    "test_img_num": 0
}
BBOX = {
    "box": [10, 10, 20, 20],
    "class": 1,
    "name": "cat",
    "score": 0.7
}
BBOXES = [BBOX]

TRAIN_DATA = {
    "img": [],
    "target": BBOXES,
    "size": [[512, 512]]
}
VALID_DATA = {
    "img": [],
    "target": BBOXES,
    "size": [[512, 512]]
}

DATASET = {
    "id": 1,
    "name": "dummy_dataset",
    "description": "dummy dataset",
    "task_id": 1,
    "ratio": 0.8,
    "class_map": CLASS_MAP,
    "class_info": CLASS_INFO,
    "train_data": TRAIN_DATA,
    "valid_data": VALID_DATA,
    "created": CREATED
}
DATASETS = [DATASET]

PARAMS_YOLOv1 = {
    "batch_size": 4,
    "imsize_h": 448,
    "imsize_w": 448,
    "load_pretrained_weight": True,
    "total_epoch": 160,
    "train_whole": False,
    "box": 2,
    "cell": 7
}
TRAIN_LOSS_LIST = [0.2, 0.1, 0.05]
VALID_LOSS_LIST = [0.25, 0.15, 0.055]

BEST_EPOCH_VALID_RESULT = {
    "IOU": 0.7,
    "mAP": 0.8,
    "loss": 0.1,
    "nth_epoch": 80,
    "prediction": [BBOXES]
}
LAST_PREDICTION_RESULT = {
    "img": [],
    "prediction": [],
    "size": []
}
RUNNING_INFO = {
    "running_state": 1,
    "total_batch": 200,
    "nth_batch": 100,
    "total_epoch": 160,
    "nth_epoch": 120,
    "last_batch_loss": 0.3
}
MODEL = {
    "id": 1,
    "task_id": 1,
    "dataset_id": 1,
    "algorithm_id": 30,
    "state": 1,
    "hyper_parameters": PARAMS_YOLOv1,
    "train_loss_list": TRAIN_LOSS_LIST,
    "valid_loss_list": VALID_LOSS_LIST,
    "best_epoch_valid_result": BEST_EPOCH_VALID_RESULT,
    "last_prediction_result": LAST_PREDICTION_RESULT,
    "running_info": RUNNING_INFO,
    "created": CREATED
}
MODELS = [MODEL]


def test_index(app):
    resp = app.get('/')
    assert resp.status_int == 200
    assert resp.headers['content-type'] == 'text/html'
    assert resp.text.strip()


def test_404(app):
    resp = app.get('/none')
    assert resp.status_int == 404


def test_create_dataset(app):
    resp = app.post('/renom_img/v2/api/detection/datasets', {})
    assert resp.status_int == 201
    assert resp.json['dataset']['id'] == DATASET['id']


def test_get_datasets(app):
    resp = app.get('/renom_img/v2/api/detection/datasets')
    assert resp.status_int == 200
    assert resp.json['datasets'][0]['id'] == DATASETS[0]['id']


def test_get_dataset(app):
    resp = app.get('/renom_img/v2/api/detection/datasets/1', {})
    assert resp.status_int == 200
    assert resp.json['dataset']['id'] == DATASET['id']


def test_update_dataset(app):
    resp = app.put('/renom_img/v2/api/detection/datasets/1', {})
    assert resp.status_int == 204


def test_delete_dataset(app):
    resp = app.delete('/renom_img/v2/api/detection/datasets/1', {})
    assert resp.status_int == 204


def test_create_model(app):
    resp = app.post('/renom_img/v2/api/detection/models', {})
    assert resp.status_int == 201
    assert resp.json['model']['id'] == MODEL['id']


def test_get_models(app):
    resp = app.get('/renom_img/v2/api/detection/models')
    assert resp.status_int == 200
    assert resp.json['models'][0]['id'] == MODELS[0]['id']


def test_get_model(app):
    resp = app.get('/renom_img/v2/api/detection/models/1')
    assert resp.status_int == 200
    assert resp.json['model']['id'] == MODEL['id']


def test_update_model(app):
    resp = app.put('/renom_img/v2/api/detection/models/1', {})
    assert resp.status_int == 204


def test_delete_model(app):
    resp = app.delete('/renom_img/v2/api/detection/models/1', {})
    assert resp.status_int == 204


def test_download_model_weight(app):
    resp = app.get('/renom_img/v2/api/detection/models/1/weight')
    pass


def test_get_train_status(app):
    resp = app.get('/renom_img/v2/api/detection/train')
    assert resp.status_int == 200
    assert resp.json['running_models'][0]['id'] == MODELS[0]['id']


def test_run_train(app):
    resp = app.post('/renom_img/v2/api/detection/train', {})
    assert resp.status_int == 201


def test_stop_train(app):
    resp = app.delete('/renom_img/v2/api/detection/train', {})
    assert resp.status_int == 204


def test_get_prediction_status(app):
    resp = app.get('/renom_img/v2/api/detection/prediction')
    assert resp.status_int == 200


def test_run_prediction(app):
    resp = app.post('/renom_img/v2/api/detection/prediction', {})
    assert resp.status_int == 201


def test_get_prediction_result(app):
    resp = app.get('/renom_img/v2/api/detection/prediction/result')
    assert resp.status_int == 200
