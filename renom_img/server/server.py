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
from PIL import Image

import numpy as np
from threading import Thread, Semaphore
from concurrent.futures import ThreadPoolExecutor as Executor
from concurrent.futures import ProcessPoolExecutor
from bottle import HTTPResponse, default_app, route, static_file, request, error

from renom.cuda import release_mem_pool
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.load import parse_txt_classification
from renom_img.api.utility.load import parse_classmap_file
from renom_img.server import wsgi_server
from renom_img.server import wsgi_server
from renom_img.server.train_thread import TrainThread
from renom_img.server.prediction_thread import PredictionThread
from renom_img.server.utility.storage import storage
from renom_img.server import State, RunningState, Task


# Thread(Future object) is stored to thread_pool as pair of "thread_id:[future, thread_obj]".
executor = Executor(max_workers=2)
train_thread_pool = {}
prediction_thread_pool = {}
respopnse_cache = {}


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
        global respopnse_cache
        try:
            ret = func(*args, **kwargs)
            if ret is None:
                ret = {}

            if respopnse_cache.get(func.__name__, None) == ret and False:
                # If server will return same value as last response, return 204.
                body = json.dumps({}, ignore_nan=True, default=json_encoder)
                return create_response(body, 204)
            else:
                assert isinstance(ret, dict),\
                    "The returned object of the API '{}' is not a dictionary.".format(func.__name__)
                respopnse_cache[func.__name__] = ret
                body = json.dumps(ret, ignore_nan=True, default=json_encoder)
                return create_response(body)

        except Exception as e:
            release_mem_pool()
            traceback.print_exc()
            body = json.dumps({"error_msg": "{}: {}".format(type(e).__name__, str(e))})
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


@route("/target/segmentation/<root:path>/<folder_name:path>/<file_name:path>")
def segmentation_target_mask(root, folder_name, file_name):
    file_dir = os.path.join('datasrc', folder_name)
    return


# WEB APIs
@route("/api/renom_img/v2/model/create", method="POST")
@json_handler
def model_create():
    req_params = request.params
    hyper_params = json.loads(req_params.hyper_params)
    hyper_params = json.loads(req_params.hyper_params)
    algorithm_id = json.loads(req_params.algorithm_id)
    dataset_id = req_params.dataset_id
    task_id = req_params.task_id
    new_id = storage.register_model(
        int(task_id), int(dataset_id), int(algorithm_id), hyper_params)
    return {"id": new_id}


@route("/api/renom_img/v2/model/load/task/<task_id:int>", method="GET")
@json_handler
def models_load_of_task(task_id):
    models = storage.fetch_models_of_task(task_id)
    # Remove best_valid_changed because it is very large.
    models = [
        {k: v if k not in [
            "best_epoch_valid_result",
            "last_prediction_result"] else {} for k, v in m.items()}
        for m in models
    ]
    return {'model_list': models}


@route("/api/renom_img/v2/model/thread/run/<id:int>", method="GET")
@json_handler
def model_thread_run(id):
    # TODO: Confirm if the model is already trained.
    thread = TrainThread(id)
    th = executor.submit(thread)
    thread.set_future(th)
    return {"status": "ok"}


@route("/api/renom_img/v2/model/thread/prediction/run/<id:int>", method="GET")
@json_handler
def model_prediction_thread_run(id):
    # TODO: Confirm if the model is already trained.
    thread = PredictionThread(id)
    th = executor.submit(thread)
    thread.set_future(th)
    return {"status": "ok"}


@route("/api/renom_img/v2/model/stop/<id:int>", method="GET")
@json_handler
def model_stop(id):
    thread = TrainThread.jobs.get(id, None)
    if thread is not None:
        thread.stop()
    return {"status": "ok"}


@route("/api/renom_img/v2/model/remove/<id:int>", method="GET")
@json_handler
def model_remove(id):
    threads = TrainThread.jobs
    active_train_thread = threads.get(id, None)
    if active_train_thread is not None:
        active_train_thread.stop()
        active_train_thread.future.result()
    storage.remove_model(id)
    return


@route("/api/renom_img/v2/model/deploy/<id:int>", method="GET")
@json_handler
def model_deploy(id):
    storage.deploy_model(id)
    return {"status": "ok"}


@route("/api/renom_img/v2/model/load/deployed/task/<id:int>", method="GET")
@json_handler
def model_load_id_deployed_of_task(id):
    dep_model = storage.fetch_deployed_model(id)
    if dep_model:
        return {"deployed_id": dep_model["id"]}
    else:
        return {"deployed_id": None}


@route("/api/renom_img/v2/model/load/best/result/<id:int>", method="GET")
@json_handler
def model_load_best_result(id):
    thread = TrainThread.jobs.get(id, None)
    if thread is None:
        saved_model = storage.fetch_model(id)
        if saved_model is None:
            return
        # If the state == STOPPED, client will never throw request.
        if saved_model["state"] != State.STOPPED.value:
            storage.update_model(id, state=State.STOPPED.value,
                                 running_state=RunningState.STOPPING.value)
            saved_model = storage.fetch_model(id)
        return {"best_result": saved_model['best_epoch_valid_result']}
    else:
        thread.returned_best_result2client()
        return {"best_result": thread.best_epoch_valid_result}


@route("/api/renom_img/v2/model/load/prediction/result/<id:int>", method="GET")
@json_handler
def model_load_prediction_result(id):
    thread = PredictionThread.jobs.get(id, None)
    if thread is None:
        saved_model = storage.fetch_model(id)
        if saved_model is None:
            raise Exception("Model id {} is not found".format(id))
        # If the state == STOPPED, client will never throw request.
        if saved_model["state"] != State.STOPPED.value:
            storage.update_model(id, state=State.STOPPED.value,
                                 running_state=RunningState.STOPPING.value)
            saved_model = storage.fetch_model(id)
        return {"result": saved_model['last_prediction_result']}
    else:
        thread.need_pull = False
        return {"result": thread.prediction_result}


@route("/api/renom_img/v2/dataset/create", method="POST")
@json_handler
def dataset_create():
    req_params = request.params
    # Receive params here.
    ratio = float(req_params.ratio)
    dataset_name = str(req_params.name)
    test_dataset_id = int(
        req_params.test_dataset_id if req_params.test_dataset_id != "undefined" else -1)
    task_id = int(req_params.task_id)
    description = str(req_params.description)
    ##

    root = pathlib.Path('datasrc')
    img_dir = root / 'img'
    label_dir = root / 'label'

    assert img_dir.exists(), \
        "The directory 'datasrc/img' is not found in current working directory."

    file_names = set([name.relative_to(img_dir) for name in img_dir.iterdir()
                      if name.is_file()])

    if test_dataset_id > 0:
        test_dataset = storage.fetch_test_dataset(test_dataset_id)
        test_dataset = set([pathlib.Path(test_path).relative_to(img_dir)
                            for test_path in test_dataset['data']['img']])

        # Remove test files.
        file_names = file_names - test_dataset

    # For Detection
    if task_id == Task.CLASSIFICATION.value:
        classification_label_dir = label_dir / "classification"
        target, class_map = parse_txt_classification(str(classification_label_dir / "target.txt"))
        target_file_list = list(target.keys())

        file_names = [p for p in file_names
                      if (img_dir / p).is_file() and (p.name in target_file_list)]

        img_files = [str(img_dir / name) for name in file_names]
        parsed_target = [target[name.name] for name in file_names]

    elif task_id == Task.DETECTION.value:
        detection_label_dir = label_dir / "detection"
        file_names = [p for p in file_names
                      if (img_dir / p).is_file() and ((detection_label_dir / p.name).with_suffix(".xml")).is_file()]
        img_files = [str(img_dir / name) for name in file_names]
        xml_files = [str(detection_label_dir / name.with_suffix('.xml')) for name in file_names]
        parsed_target, class_map = parse_xml_detection(xml_files, num_thread=8)
    elif task_id == Task.SEGMENTATION.value:
        segmentation_label_dir = label_dir / "segmentation"
        file_names = [p for p in file_names if (img_dir / p).is_file() and
                      any([((segmentation_label_dir / p.name).with_suffix(suf)).is_file()
                           for suf in [".jpg", ".png"]])]
        img_files = [str(img_dir / name) for name in file_names]
        parsed_target = [str(segmentation_label_dir / name.with_suffix(".png"))
                         for name in file_names]
        class_map = parse_classmap_file(str(segmentation_label_dir / "class_map.txt"))
        print(class_map)

    # Split into train and valid.
    n_imgs = len(file_names)
    perm = np.random.permutation(n_imgs)

    train_img, valid_img = np.split(np.array([img_files[index] for index in perm]),
                                    [int(ratio * n_imgs)])
    train_img = train_img.tolist()
    valid_img = valid_img.tolist()
    valid_img_size = [list(Image.open(i).size) for i in valid_img]

    train_target, valid_target = np.split(np.array([parsed_target[index] for index in perm]),
                                          [int(ratio * n_imgs)])
    train_target = train_target.tolist()
    valid_target = valid_target.tolist()

    # Register
    train_data = {
        'img': train_img,
        'target': train_target
    }
    valid_data = {
        'img': valid_img,
        'target': valid_target,
        'size': valid_img_size,
    }
    dataset_id = storage.register_dataset(task_id, dataset_name, description, ratio,
                                          train_data, valid_data,
                                          class_map, {}, test_dataset_id
                                          )
    return {
        'id': dataset_id,
        'valid_data': valid_data,
        'class_map': class_map,
    }


@route("/api/renom_img/v2/dataset/load/task/<id:int>", method="GET")
@json_handler
def dataset_load_of_task(id):
    # TODO: Remember last sent value and cache it.
    datasets = storage.fetch_datasets_of_task(id)
    return {
        "dataset_list": [
            {
                'id': d["id"],
                'name': d["name"],
                'class_map': d["class_map"],
                'task_id': d["task_id"],
                'valid_data': d["valid_data"],
                'ratio': d["ratio"],
                'description': d["description"],
                'test_dataset_id': d["test_dataset_id"],
            }
            for d in datasets
        ]
    }


@route("/api/renom_img/v2/test_dataset/create", method="POST")
@json_handler
def test_dataset_create():
    req_params = request.params
    # Receive params here.
    ratio = float(req_params.ratio)
    dataset_name = str(req_params.name)
    task_id = int(req_params.task_id)
    description = str(req_params.description)
    ##
    root = pathlib.Path('datasrc')
    img_dir = root / 'img'
    label_dir = root / 'label'

    assert img_dir.exists(), \
        "The directory 'datasrc/img' is not found in current working directory."
    assert label_dir.exists(), \
        "The directory 'datasrc/label/detection' is not found in current working directory."

    file_names = [name.relative_to(img_dir) for name in img_dir.iterdir()
                  if name.is_file()]

    # For Detection
    if task_id == Task.CLASSIFICATION.value:
        classification_label_dir = label_dir / "classification"
        target, class_map = parse_txt_classification(str(classification_label_dir / "target.txt"))
        target_file_list = list(target.keys())

        file_names = [p for p in file_names
                      if (img_dir / p).is_file() and (p.name in target_file_list)]

        n_imgs = len(file_names)
        perm = np.random.permutation(n_imgs)
        file_names = [file_names[index] for index in perm[:int(n_imgs * ratio)]]

        img_files = [str(img_dir / name) for name in file_names]
        parsed_target = [target[name.name] for name in file_names]

    elif task_id == Task.DETECTION.value:
        detection_label_dir = label_dir / "detection"
        file_names = [p for p in file_names
                      if (img_dir / p).is_file() and ((detection_label_dir / p.name).with_suffix(".xml")).is_file()]
        n_imgs = len(file_names)
        perm = np.random.permutation(n_imgs)
        file_names = [file_names[index] for index in perm[:int(n_imgs * ratio)]]

        img_files = [str(img_dir / name) for name in file_names]
        xml_files = [str(detection_label_dir / name.with_suffix('.xml')) for name in file_names]
        parsed_target, class_map = parse_xml_detection(xml_files, num_thread=8)
    elif task_id == Task.SEGMENTATION.value:
        pass

    test_data = {
        "img": img_files,
        "target": parsed_target,
    }
    test_dataset_id = storage.register_test_dataset(task_id, dataset_name, description, test_data)
    return {
        'id': test_dataset_id,
        'test_data': test_data,
        'class_map': class_map,
    }


@route("/api/renom_img/v2/polling/train/model/<id:int>", method="GET")
@json_handler
def polling_train(id):
    """

    Cations: 
        This function is possible to return empty dictionary.
    """
    threads = TrainThread.jobs
    active_train_thread = threads.get(id, None)
    if active_train_thread is None:
        saved_model = storage.fetch_model(id)
        if saved_model is None:
            return

        # If the state == STOPPED, client will never throw request.
        if saved_model["state"] != State.STOPPED.value:
            storage.update_model(id, state=State.STOPPED.value,
                                 running_state=RunningState.STOPPING.value)
            saved_model = storage.fetch_model(id)

        return {
            "state": saved_model["state"],
            "running_state": saved_model["running_state"],
            "total_epoch": saved_model["total_epoch"],
            "nth_epoch": saved_model["nth_epoch"],
            "total_batch": saved_model["total_batch"],
            "nth_batch": saved_model["nth_batch"],
            "last_batch_loss": saved_model["last_batch_loss"],
            "total_valid_batch": 0,
            "nth_valid_batch": 0,
            "best_result_changed": False,
            "train_loss_list": saved_model["train_loss_list"],
            "valid_loss_list": saved_model["valid_loss_list"],
        }
    elif active_train_thread.state == State.RESERVED or \
            active_train_thread.state == State.CREATED:

        for _ in range(60):
            if active_train_thread.state == State.RESERVED or \
                    active_train_thread.state == State.CREATED:
                time.sleep(1)
                if active_train_thread.updated:
                    active_train_thread.returned2client()
                    break
            else:
                time.sleep(1)
                break

        active_train_thread.consume_error()
        return {
            "state": active_train_thread.state.value,
            "running_state": active_train_thread.running_state.value,
            "total_epoch": 0,
            "nth_epoch": 0,
            "total_batch": 0,
            "nth_batch": 0,
            "last_batch_loss": 0,
            "total_valid_batch": 0,
            "nth_valid_batch": 0,
            "best_result_changed": False,
            "train_loss_list": [],
            "valid_loss_list": [],
        }
    else:
        for _ in range(10):
            time.sleep(0.5)  # Avoid many request.
            if active_train_thread.updated:
                break
            active_train_thread.consume_error()
        active_train_thread.returned2client()
        return {
            "state": active_train_thread.state.value,
            "running_state": active_train_thread.running_state.value,
            "total_epoch": active_train_thread.total_epoch,
            "nth_epoch": active_train_thread.nth_epoch,
            "total_batch": active_train_thread.total_batch,
            "nth_batch": active_train_thread.nth_batch,
            "last_batch_loss": active_train_thread.last_batch_loss,
            "total_valid_batch": 0,
            "nth_valid_batch": 0,
            "best_result_changed": active_train_thread.best_valid_changed,
            "train_loss_list": active_train_thread.train_loss_list,
            "valid_loss_list": active_train_thread.valid_loss_list,
        }


@route("/api/renom_img/v2/polling/prediction/model/<id:int>", method="GET")
@json_handler
def polling_prediction(id):
    """
    Cations: 
        This function is possible to return empty dictionary.
    """
    threads = PredictionThread.jobs
    active_prediction_thread = threads.get(id, None)
    if active_prediction_thread is None:
        time.sleep(0.5)  # Avoid many request.
        return {
            "need_pull": False,
            "state": State.STOPPED.value,
            "running_state": RunningState.STOPPING.value,
            "total_batch": 0,
            "nth_batch": 0,
        }
    elif active_prediction_thread.state == State.PRED_RESERVED or \
            active_prediction_thread.state == State.PRED_CREATED:
        time.sleep(0.5)  # Avoid many request.
        return {
            "need_pull": active_prediction_thread.need_pull,
            "state": active_prediction_thread.state.value,
            "running_state": active_prediction_thread.running_state.value,
            "total_batch": active_prediction_thread.total_batch,
            "nth_batch": active_prediction_thread.nth_batch,
        }
    else:
        for _ in range(10):
            time.sleep(0.5)  # Avoid many request.
            if active_prediction_thread.updated:
                break
            active_prediction_thread.consume_error()
        active_prediction_thread.returned2client()
        return {
            "need_pull": active_prediction_thread.need_pull,
            "state": active_prediction_thread.state.value,
            "running_state": active_prediction_thread.running_state.value,
            "total_batch": active_prediction_thread.total_batch,
            "nth_batch": active_prediction_thread.nth_batch,
        }


@route("/api/renom_img/v2/test_dataset/load/task/<id:int>", method="GET")
@json_handler
def test_dataset_load_of_task(id):
    # TODO: Remember last sent value and cache it.
    datasets = storage.fetch_test_datasets_of_task(id)
    return {
        "test_dataset_list": datasets
    }


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
