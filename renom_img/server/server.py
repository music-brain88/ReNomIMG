# coding: utf-8

import os
import time
import base64
import pkg_resources
import argparse
import urllib
import mimetypes
import posixpath
import traceback
import pathlib
import random
from datetime import datetime
from collections import OrderedDict
import xmltodict
import simplejson as json

import numpy as np
from threading import Thread
from bottle import HTTPResponse, default_app, route, static_file, request, error

from renom.cuda import release_mem_pool
from renom_img.api.utility.load import parse_xml_detection
from renom_img.server import wsgi_server
from renom_img.server.train_thread import TrainThread
from renom_img.server.prediction_thread import PredictionThread
from renom_img.server.utility.storage import storage
from renom_img.server import State, RunningState


# Thread(Future object) is stored to thread_pool as pair of "thread_id:[future, thread_obj]".
train_thread_pool = {}
prediction_thread_pool = {}


def get_train_thread_count():
    return len([th for th in train_thread_pool.values() if th[0].running()])


def create_response(body, status=200):
    r = HTTPResponse(status=status, body=body)
    r.set_header('Content-Type', 'application/json')
    return r


def strip_path(filename):
    if os.path.isabs(filename):
        raise ValueError('Invalid path')
    if '..' in filename:
        raise ValueError('Invalid path')
    if ':' in filename:
        raise ValueError('Invalid path')

    filename = filename.strip().strip('./\\')
    return filename


def _get_resource(path, filename):
    filename = strip_path(filename)
    body = pkg_resources.resource_string(__name__, posixpath.join('.build', path, filename))

    headers = {}
    mimetype, encoding = mimetypes.guess_type(filename)
    if mimetype:
        headers['Content-Type'] = mimetype
    if encoding:
        headers['encoding'] = encoding
    return HTTPResponse(body, **headers)


def json_encoder(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()


def json_handler(func):
    def wrapped(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
            if ret is None:
                ret = {}
            assert isinstance(ret, dict),\
                "The returned object of the API '{}' is not a dictionary.".format(func.__name__)
            body = json.dumps(ret, ignore_nan=True, default=json_encoder)
            return create_response(body)
        except Exception as e:
            release_mem_pool()
            traceback.print_exc()
            body = json.dumps({"error_msg": str(e)})
            ret = create_response(body, 500)
            return ret
    return wrapped


@route("/")
def index():
    return _get_resource('', 'index.html')


@route("/static/<file_name:re:.+>")
def static(file_name):
    return _get_resource('static', file_name)


@route("/css/<file_name:path>")
def css(file_name):
    return _get_resource('static/css/', file_name)


@route("/fonts/<file_name:path>")
def font(file_name):
    return _get_resource('static/fonts/', file_name)


@error(404)
def error404(error):
    body = json.dumps({"error_msg": "Page Not Found"})
    ret = create_response(body, status=404)
    return ret


@route("/datasrc/<folder_name:path>/<file_name:path>")
def datasrc(folder_name, file_name):
    file_dir = os.path.join('datasrc', folder_name)
    return static_file(file_name, root=file_dir, mimetype='image/*')


# WEB APIs
@route("/api/renom_img/v2/model/create", method="POST")
@json_handler
def model_create():
    req_params = request.params
    hyper_params = json.loads(req_params.hyper_params)
    algorithm_id = json.loads(req_params.algorithm_id)
    parents = json.loads(req_params.parents)
    dataset_id = req_params.dataset_id
    task_id = req_params.task_id
    new_id = storage.register_model(
        int(task_id), int(dataset_id), int(algorithm_id), hyper_params)
    return {"id": new_id}


@route("/api/renom_img/v2/model/load/<id:int>", method="GET")
@json_handler
def model_load(id):
    model = storage.fetch_model(id)
    return {"model": model}


@route("/api/renom_img/v2/model/load/task/<task_id:int>", method="GET")
@json_handler
def models_load_of_task(task_id):
    models = storage.fetch_models_of_task(task_id)
    print(models)
    return {'model_list': models}


@route("/api/renom_img/v2/model/run/<id:int>", method="GET")
@json_handler
def model_run(id):
    thread = TrainThread(id)
    thread.run()
    return


@route("/api/renom_img/v2/model/remove/<id:int>", method="GET")
@json_handler
def model_remove(id):
    return


@route("/api/renom_img/v2/load_deployed_models", method="GET")
@json_handler
def models_load_deployed():
    dep_models = storage.fetch_task()
    json = {"models": dep_models}
    return json


@route("/api/renom_img/v2/dataset/create", method="POST")
@json_handler
def dataset_create():
    # req_params = request.params
    # Receive params here.
    ratio = 0.8
    dataset_name = "test"
    test_dataset_id = 1
    task_id = 1
    description = "This is test."
    ##

    root = pathlib.Path('datasrc')
    img_dir = root / 'img'
    label_dir = root / 'label' / 'detection'

    assert img_dir.exists(), \
        "The directory 'datasrc/img' is not found in current working directory."
    assert label_dir.exists(), \
        "The directory 'datasrc/label/detection' is not found in current working directory."

    file_names = set([name.relative_to(img_dir) for name in img_dir.iterdir()
                      if name.is_file() and (label_dir / name.relative_to(img_dir)).with_suffix('.xml').is_file()])

    test_dataset = storage.fetch_test_dataset(test_dataset_id)
    test_dataset = set([pathlib.Path(test_path).relative_to(img_dir)
                        for test_path in test_dataset['data']['img']])

    # Remove test files.
    file_names = file_names - test_dataset

    # Parse
    img_files = [str(img_dir / name) for name in file_names]
    xml_files = [str(label_dir / name.with_suffix('.xml')) for name in file_names]
    parsed_xml, class_map = parse_xml_detection(xml_files, num_thread=8)

    # Split into train and valid.
    n_imgs = len(file_names)
    perm = np.random.permutation(n_imgs)

    train_img, valid_img = np.split(np.array([img_files[index] for index in perm]),
                                    [int(ratio * n_imgs)])
    train_img = train_img.tolist()
    valid_img = valid_img.tolist()

    train_xml, valid_xml = np.split(np.array([parsed_xml[index] for index in perm]),
                                    [int(ratio * n_imgs)])
    train_xml = train_xml.tolist()
    valid_xml = valid_xml.tolist()

    # Register
    dataset_id = storage.register_dataset(task_id, dataset_name, description, ratio,
                                          {'img': train_img, 'target': train_xml}, {
                                              'img': valid_img, 'target': valid_xml},
                                          class_map, {}, test_dataset_id
                                          )
    return {'id': dataset_id}


@route("/api/renom_img/v2/test_dataset/create", method="POST")
@json_handler
def test_dataset_create():
    # req_params = request.params
    # Receive params here.
    ratio = 0.1
    dataset_name = "test"
    task_id = 1
    description = "This is test."
    ##
    root = pathlib.Path('datasrc')
    img_dir = root / 'img'
    label_dir = root / 'label' / 'detection'

    assert img_dir.exists(), \
        "The directory 'datasrc/img' is not found in current working directory."
    assert label_dir.exists(), \
        "The directory 'datasrc/label/detection' is not found in current working directory."

    file_names = [name.relative_to(img_dir) for name in img_dir.iterdir()
                  if name.is_file() and (label_dir / name.relative_to(img_dir)).with_suffix('.xml').is_file()]

    n_imgs = len(file_names)
    perm = np.random.permutation(n_imgs)
    file_names = [file_names[index] for index in perm[:int(n_imgs * ratio)]]

    img_files = [str(img_dir / name) for name in file_names]
    xml_files = [str(label_dir / name.with_suffix('.xml')) for name in file_names]
    parsed_xml, class_map = parse_xml_detection(xml_files, num_thread=8)

    test_dataset_id = storage.register_test_dataset(task_id, dataset_name, description, {
        "img": img_files,
        "target": parsed_xml
    })
    return {"id": test_dataset_id}


@route("/api/renom_img/v2/polling/progress/model/<id:int>", method="GET")
@json_handler
def polling_progress(id):
    return


@route("/api/renom_img/v2/polling/validation/model/<id:int>", method="GET")
@json_handler
def polling_validation(id):
    return


@route("/api/renom_img/v2/polling/weight/download", method="GET")
@json_handler
def polling_weight_download():
    return


@route("/api/renom_img/v2/polling/prediction", method="GET")
@json_handler
def polling_prediction():
    return


def main():
    # Parser settings.
    parser = argparse.ArgumentParser(description='ReNomIMG')
    parser.add_argument('--host', default='0.0.0.0', help='Server address')
    parser.add_argument('--port', default='8080', help='Server port')
    args = parser.parse_args()
    wsgiapp = default_app()
    httpd = wsgi_server.Server(wsgiapp, host=args.host, port=int(args.port))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
