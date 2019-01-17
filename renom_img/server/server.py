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
from renom_img.api.utility.load import parse_image_segmentation
from renom_img.server import wsgi_server
from renom_img.server import wsgi_server
from renom_img.server.train_thread import TrainThread
from renom_img.server.prediction_thread import PredictionThread
from renom_img.server.utility.storage import storage
from renom_img.server import State, RunningState, Task
from renom_img.server import DATASET_IMG_DIR, DATASET_LABEL_CLASSIFICATION_DIR, \
    DATASET_LABEL_DETECTION_DIR, DATASET_LABEL_SEGMENTATION_DIR
from renom_img.server.utility.setup_example import setup_example


# Thread(Future object) is stored to thread_pool as pair of "thread_id:[future, thread_obj]".
executor = Executor(max_workers=2)
train_thread_pool = {}
prediction_thread_pool = {}
respopnse_cache = {}

# temporary stored dataset
# {hash: dataset}
temp_dataset = {}


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


@route("/api/target/segmentation", method="POST")
@json_handler
def segmentation_target_mask():
    size = json.loads(request.params.size)
    path = request.params.name
    path = pathlib.Path(path)
    file_dir = path.with_suffix('.png').relative_to('datasrc/img')
    file_dir = 'datasrc/label/segmentation' / file_dir
    img = np.array(Image.open(file_dir).resize(size)).astype(np.uint8)
    img[img == 255] = 0
    img = img.tolist()
    return {"class": img}


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


@route("/api/renom_img/v2/model/undeploy/<task_id:int>", method="GET")
@json_handler
def model_undeploy(task_id):
    storage.undeploy_model(task_id)
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


@route("/api/renom_img/v2/dataset/confirm", method="POST")
@json_handler
def dataset_confirm():
    global temp_dataset
    req_params = request.params
    dataset_hash = req_params.hash
    # Receive params here.
    ratio = float(req_params.ratio)
    # Correspondence for multi byte input data
    dataset_name = str(urllib.parse.unquote(req_params.name, encoding='utf-8'))
    test_dataset_id = int(
        req_params.test_dataset_id
        if req_params.test_dataset_id != '' else '-1')
    task_id = int(req_params.task_id)
    description = str(urllib.parse.unquote(req_params.description, encoding='utf-8'))
    #

    root = pathlib.Path('datasrc')
    img_dir = root / 'img'
    label_dir = root / 'label'

    assert img_dir.exists(), \
        "The directory 'datasrc/img' is\
         not found in current working directory."

    file_names = set([name.relative_to(img_dir) for name in img_dir.iterdir()
                      if name.is_file()])

    if test_dataset_id > 0:
        test_dataset = storage.fetch_test_dataset(test_dataset_id)
        test_dataset = set([pathlib.Path(test_path).relative_to(img_dir)
                            for test_path in test_dataset['data']['img']])

        # Remove test files.
        file_names = file_names - test_dataset
    import time
    # For Detection
    if task_id == Task.CLASSIFICATION.value:

        start_t = time.time()
        classification_label_dir = DATASET_LABEL_CLASSIFICATION_DIR
        target, class_map = parse_txt_classification(str(classification_label_dir / "target.txt"))
        target_file_list = list(target.keys())

        file_names = [p for p in file_names
                      if (img_dir / p).is_file() and (p.name in target_file_list)]

        img_files = [str(img_dir / name) for name in file_names]
        parsed_target = [target[name.name] for name in file_names]

    elif task_id == Task.DETECTION.value:
        detection_label_dir = DATASET_LABEL_DETECTION_DIR
        file_names = [p for p in file_names
                      if (img_dir / p).is_file() and ((detection_label_dir / p.name).with_suffix(".xml")).is_file()]
        img_files = [str(img_dir / name) for name in file_names]
        xml_files = [str(detection_label_dir / name.with_suffix('.xml')) for name in file_names]
        parsed_target, class_map = parse_xml_detection(xml_files, num_thread=8)

    elif task_id == Task.SEGMENTATION.value:
        segmentation_label_dir = DATASET_LABEL_SEGMENTATION_DIR
        file_names = [p for p in file_names if (img_dir / p).is_file() and
                      any([((segmentation_label_dir / p.name).with_suffix(suf)).is_file()
                           for suf in [".jpg", ".png"]])]
        img_files = [str(img_dir / name) for name in file_names]
        parsed_target = [str(segmentation_label_dir / name.with_suffix(".png"))
                         for name in file_names]
        class_map = parse_classmap_file(str(segmentation_label_dir / "class_map.txt"))

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

    # Load test Dataset if exists.
    if test_dataset_id > 0:
        test_dataset = storage.fetch_test_dataset(test_dataset_id)
        test_ratio = []
    else:
        test_ratio = []

    # Dataset Information
    if task_id == Task.CLASSIFICATION.value:

        start_t = time.time()
        train_tag_num, _ = np.histogram(train_target, bins=list(range(len(class_map) + 1)))
        valid_tag_num, _ = np.histogram(valid_target, bins=list(range(len(class_map) + 1)))

    elif task_id == Task.DETECTION.value:
        train_tag_list = []
        valid_tag_list = []

        for i in range(len(train_target)):
            train_tag_list.append(train_target[i][0].get('class'))

        for i in range(len(valid_target)):
            valid_tag_list.append(valid_target[i][0].get('class'))

        train_tag_num, _ = np.histogram(train_tag_list, bins=list(range(len(class_map) + 1)))
        valid_tag_num, _ = np.histogram(valid_tag_list, bins=list(range(len(class_map) + 1)))
    elif task_id == Task.SEGMENTATION.value:
        train_tag_num = parse_image_segmentation(train_target, len(class_map), 8)
        valid_tag_num = parse_image_segmentation(valid_target, len(class_map), 8)

    class_info = {
        "class_map": class_map,
        "class_ratio": ((train_tag_num + valid_tag_num) / np.sum(train_tag_num + valid_tag_num)).tolist(),
        "train_ratio": (train_tag_num / (train_tag_num + valid_tag_num)).tolist(),
        "valid_ratio": (valid_tag_num / (train_tag_num + valid_tag_num)).tolist(),
        "test_ratio": test_ratio,
        "train_img_num": len(train_img),
        "valid_img_num": len(valid_img),
        "test_img_num": 1,
    }

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

    dataset = {
        "task_id": task_id,
        "dataset_name": dataset_name,
        "description": description,
        "ratio": ratio,
        "train_data": train_data,
        "valid_data": valid_data,
        "class_map": class_map,
        "test_dataset_id": test_dataset_id,
        "class_info": class_info
    }
    temp_dataset[dataset_hash] = dataset

    # Client doesn't need 'train_data'
    return_dataset = {
        "task_id": task_id,
        "dataset_name": dataset_name,
        "description": description,
        "ratio": ratio,
        # "train_data": train_data,
        "valid_data": valid_data,
        "class_map": class_map,
        "test_dataset_id": test_dataset_id,
        "class_info": class_info
    }
    return return_dataset


@route("/api/renom_img/v2/dataset/create", method="POST")
@json_handler
def dataset_create():
    """
    dataset_confirm => dataset_create.

    Note: User can rename the dataset name and description.
    If the hash equals to the dataset in the temp_dataset but
    the requested ratio and saved ratio is not same, raise Exception.

    """
    global temp_dataset

    # Request params.
    req_params = request.params
    dataset_hash = req_params.hash
    request_dataset_name = str(urllib.parse.unquote(req_params.name, encoding='utf-8'))
    request_dataset_description = str(urllib.parse.unquote(
        req_params.description, encoding='utf-8'))
    request_dataset_ratio = float(req_params.ratio)
    request_test_dataset_id = int(req_params.test_dataset_id)

    # Saved params.
    dataset = temp_dataset.get(dataset_hash, False)
    assert dataset, "Requested dataset does not exist."

    task_id = dataset.get('task_id')
    dataset_name = dataset.get('dataset_name')
    description = dataset.get('description')
    ratio = dataset.get('ratio')
    train_data = dataset.get('train_data')
    valid_data = dataset.get('valid_data')
    class_map = dataset.get('class_map')
    class_info = dataset.get('class_info')
    test_dataset_id = dataset.get('test_dataset_id')

    assert ratio == request_dataset_ratio, \
        "Requested ratio and saved one is not same. {}, {}".format(ratio, request_dataset_ratio)
    assert test_dataset_id == request_test_dataset_id, "Requested test_dataset_id and saved one is not same."

    dataset_id = storage.register_dataset(
        task_id,
        dataset_name,
        description,
        ratio,
        train_data,
        valid_data,
        class_map,
        class_info,
        test_dataset_id
    )

    temp_dataset = {}  # Release
    return {'dataset_id': dataset_id}


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
                'class_info': d["class_info"],
            }
            for d in datasets
        ]
    }


@route("/api/renom_img/v2/test_dataset/confirm", method="POST")
@json_handler
def test_dataset_confirm():
    req_params = request.params
    # Receive params here.
    ratio = float(req_params.ratio)
    dataset_name = str(urllib.parse.unquote(req_params.name, encoding='utf-8'))
    task_id = int(req_params.task_id)
    description = str(urllib.parse.unquote(req_params.description, encoding='utf-8'))
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

    class_map = {}
    n_imgs = 0
    # For Detection
    if task_id == Task.CLASSIFICATION.value:
        classification_label_dir = DATASET_LABEL_CLASSIFICATION_DIR
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
        detection_label_dir = DATASET_LABEL_DETECTION_DIR
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
    # test_dataset_id = storage.register_test_dataset(task_id, dataset_name, description, test_data)
    test_tag_num = []
    if task_id == Task.DETECTION.value:
        test_tag_list = []

        for i in range(len(parsed_target)):
            test_tag_list.append(parsed_target[i][0].get('class'))

        test_tag_num, _ = np.histogram(test_tag_list, bins=list(range(len(class_map) + 1)))

    class_info = {
        "test_dataset_name": dataset_name,
        "class_map": class_map,
        "other_imgs": (n_imgs - len(img_files)),
        "test_imgs": len(img_files),
        "class_ratio": test_tag_num.tolist(),
        "test_ratio": ratio,
    }

    #   "class_ratio": ((train_tag_num + valid_tag_num) / np.sum(train_tag_num + valid_tag_num)).tolist(),
    #   "train_ratio": (train_tag_num / (train_tag_num + valid_tag_num)).tolist(),
    #   "valid_ratio": (valid_tag_num / (train_tag_num + valid_tag_num)).tolist(),
    #   "test_ratio": test_ratio,
    # }

    return class_info


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
        classification_label_dir = DATASET_LABEL_CLASSIFICATION_DIR
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
        detection_label_dir = DATASET_LABEL_DETECTION_DIR
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


@route("/api/renom_img/v2/deployed_model/task/<task_id:int>", method="GET")
def pull_deployed_model(task_id):
    # This method will be called from python script.
    try:
        ret = storage.fetch_deployed_model(task_id)
        if ret is None:
            raise Exception("No model deployed.")
        file_name = ret['best_epoch_weight']
        return static_file(file_name, root=".", download='deployed_model.h5')
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v2/deployed_model_info/task/<task_id:int>", method="GET")
@json_handler
def get_deployed_model_info(task_id):
    # This method will be called from python script.
    saved_model = storage.fetch_deployed_model(task_id)
    if saved_model is None:
        raise Exception("No model deployed.")

    return {
        "state": saved_model["state"],
        "running_state": saved_model["running_state"],
        "total_epoch": saved_model["total_epoch"],
        "algorithm_id": saved_model["algorithm_id"],
        "hyper_parameters": saved_model["hyper_parameters"],
        "filename": saved_model["best_epoch_weight"],
    }


def main():
    # Parser settings.
    parser = argparse.ArgumentParser(description='ReNomIMG')
    parser.add_argument('--host', default='0.0.0.0', help='Server address')
    parser.add_argument('--port', default='8080', help='Server port')
    subparsers = parser.add_subparsers()
    parser_add = subparsers.add_parser('setup_example', help='Setup example dataset.')
    parser_add.set_defaults(handler=setup_example)
    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler()
        return
    wsgiapp = default_app()
    httpd = wsgi_server.Server(wsgiapp, host=args.host, port=int(args.port))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
