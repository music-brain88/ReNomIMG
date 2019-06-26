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
import pandas as pd
from datetime import datetime
from collections import OrderedDict
import xmltodict
import simplejson as json
from PIL import Image

from pandas.io.json import json_normalize
import numpy as np
from threading import Thread, Semaphore
from concurrent.futures import ThreadPoolExecutor as Executor
from concurrent.futures import ProcessPoolExecutor
from bottle import HTTPResponse, default_app, route, static_file, request, error

from sqlalchemy.exc import SQLAlchemyError

from renom.cuda import release_mem_pool
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.load import parse_txt_classification
from renom_img.api.utility.load import parse_classmap_file
from renom_img.api.utility.load import parse_image_segmentation
from renom_img.server import wsgi_server
from renom_img.server.train_thread import TrainThread
from renom_img.server.prediction_thread import PredictionThread
from renom_img.server.utility.storage import storage
from renom_img.server import Algorithm, State, RunningState, Task
from renom_img.server import DATASET_IMG_DIR, DATASET_LABEL_CLASSIFICATION_DIR, \
    DATASET_LABEL_DETECTION_DIR, DATASET_LABEL_SEGMENTATION_DIR
from renom_img.server import DATASET_NAME_MAX_LENGTH, DATASET_NAME_MIN_LENGTH, \
    DATASET_DESCRIPTION_MAX_LENGTH, DATASET_DESCRIPTION_MIN_LENGTH, \
    DATASET_RATIO_MAX, DATASET_RATIO_MIN, EPOCH_MAX, EPOCH_MIN, \
    BATCH_MAX, BATCH_MIN, TASK_ID_BY_NAME, ERROR_MESSAGE_TEMPLATE

from renom_img.server.utility.setup_example import setup_example
from renom_img.server.utility.formatter import get_formatter_resolver

from renom_img.server.utility.error import ReNomIMGServerError, ForbiddenError, NotFoundError, \
    MethodNotAllowedError, ServiceUnavailableError, \
    InvalidRequestParamError, DatasetNotFoundError, ModelNotFoundError, WeightNotFoundError, \
    ModelRunningError, MemoryOverflowError, DirectoryNotFound, TaskNotFoundError


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


def dump_json(data):
    return json.dumps(data, ignore_nan=True, default=json_encoder)


def create_response(body, status=200):
    r = HTTPResponse(status=status, body=dump_json(body))
    r.set_header('Content-Type', 'application/json')
    return r


def create_error_response(error, status=500):
    if not isinstance(error, ReNomIMGServerError):
        error = ReNomIMGServerError("Unkown error occured.")
    body = {"error": {"code": error.code, "message": error.message}}
    return create_response(body, status=status)


def logging_error(error):
    # error logging operation
    traceback.print_exc()


def is_int(s):
    try:
        int(s)
    except:
        return False
    return True


def is_float(s):
    try:
        float(s)
    except:
        return False
    return True


def get_task_id_by_name(task_name):
    task_exists(task_name)
    return TASK_ID_BY_NAME[task_name]


def task_exists(task_name):
    if task_name not in TASK_ID_BY_NAME.keys():
        raise TaskNotFoundError("Task {} is not available. Please select from {}.".format(task_name, list(TASK_ID_BY_NAME.keys())))


def check_dataset_exists(dataset, dataset_id):
    if dataset is None:
        raise DatasetNotFoundError("Dataset {} is not found.".format(dataset_id))


def check_model_exists(model, model_id):
    if model is None:
        raise ModelNotFoundError("Model {} is not found.".format(model_id))


def check_weight_exists(filepath, model_id):
    if os.path.isfile(filepath):
        raise WeightNotFoundError("Model {} is exists, but weight file is not found.".format(model_id))


def check_model_running(model_id):
    thread = TrainThread.jobs.get(model_id, None)
    if thread is not None:
        raise ModelRunningError("Model {} is running, please wait or stop train.")


def error_message_dataset_name(name):
    message = ""
    param_name = "Dataset name"
    if name is None:
        desc = "is required"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, DATASET_NAME_MIN_LENGTH, DATASET_NAME_MAX_LENGTH)
    elif len(name) < DATASET_NAME_MIN_LENGTH:
        desc = "too short"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, DATASET_NAME_MIN_LENGTH, DATASET_NAME_MAX_LENGTH)
    elif len(name) > DATASET_NAME_MAX_LENGTH:
        desc = "too long"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, DATASET_NAME_MIN_LENGTH, DATASET_NAME_MAX_LENGTH)
    return message


def error_message_dataset_ratio(ratio):
    message = ""
    param_name = "Dataset ratio"
    if ratio is None:
        desc = "is required"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, DATASET_RATIO_MIN, DATASET_RATIO_MAX)
    elif not is_float(ratio):
        desc = "must be float value"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, DATASET_RATIO_MIN, DATASET_RATIO_MAX)
    elif float(ratio) < DATASET_RATIO_MIN:
        desc = "is too small"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, DATASET_RATIO_MIN, DATASET_RATIO_MAX)
    elif float(ratio) > DATASET_RATIO_MAX:
        desc = "is too large"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, DATASET_RATIO_MIN, DATASET_RATIO_MAX)
    return message


def error_message_dataset_desc(desc):
    message = ""
    param_name = "Dataset description"
    if desc is None:
        return message
    if len(desc) < DATASET_DESCRIPTION_MIN_LENGTH:
        desc = "too short"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, DATASET_DESCRIPTION_MIN_LENGTH, DATASET_DESCRIPTION_MAX_LENGTH)
    elif len(desc) > DATASET_DESCRIPTION_MAX_LENGTH:
        desc = "too long"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, DATASET_DESCRIPTION_MIN_LENGTH, DATASET_DESCRIPTION_MAX_LENGTH)
    return message


def check_create_dataset_params(params):
    messages = []
    # check params has name.
    name = params.get("name", None)
    m = error_message_dataset_name(name)
    if len(m) > 0:
        messages.append(m)

    # check ratio
    ratio = params.get("ratio", None)
    m = error_message_dataset_ratio(ratio)
    if len(m) > 0:
        messages.append(m)

    # check description
    description = params.get("description", None)
    m = error_message_dataset_desc(description)
    if len(m) > 0:
        messages.append(m)

    if len(messages) > 0:
        raise InvalidRequestParamError("\n".join(messages))


def error_message_model_dataset_id(dataset_id):
    message = ""
    if dataset_id is None:
        message = "Dataset id is required. Please input exists dataset id."
    elif not is_int(dataset_id):
        message = "Dataset id is must be integer."
    return message


def error_message_model_algorithm_id(algorithm_id):
    message = ""
    if algorithm_id is None:
        message = "Algorithm id is required. Please select from {}.".format([{a.name: a.value} for a in Algorithm])
    elif not is_int(algorithm_id):
        message = "Algorithm id is must be integer. Please select from {}.".format([{a.name: a.value} for a in Algorithm])
    elif is_int(algorithm_id) not in [a.value for a in Algorithm]:
        message = "Algorithm id is not exists. Please select from {}.".format([{a.name: a.value} for a in Algorithm])
    return message


def error_message_epoch(epoch):
    message = ""
    param_name = "Hyperparameter epoch"
    if epoch is None:
        desc = "is required"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, EPOCH_MIN, EPOCH_MAX)
    elif not is_int(epoch):
        desc = "must be integer"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, EPOCH_MIN, EPOCH_MAX)
    elif is_int(epoch) < EPOCH_MIN:
        desc = "is too small"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, EPOCH_MIN, EPOCH_MAX)
    elif is_int(epoch) > EPOCH_MAX:
        desc = "is too large"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, EPOCH_MIN, EPOCH_MAX)
    return message


def error_message_batch(batch):
    message = ""
    param_name = "Hyperparameter batch"
    if batch is None:
        desc = "is required"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, BATCH_MIN, BATCH_MAX)
    elif not is_int(batch):
        desc = "must be integer"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, BATCH_MIN, BATCH_MAX)
    elif is_int(batch) < BATCH_MIN:
        desc = "is too small"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, BATCH_MIN, BATCH_MAX)
    elif is_int(batch) > BATCH_MAX:
        desc = "is too large"
        message = ERROR_MESSAGE_TEMPLATE.format(param_name, desc, BATCH_MIN, BATCH_MAX)
    return message


def error_message_model_hyper_params(hyper_params):
    messages = []
    if hyper_params is None:
        messages.append("Hyper parameter is not exists.")
    else:
        print()
        # check epoch
        m = error_message_epoch(hyper_params["total_epoch"])
        if len(m) > 0:
            messages.append(m)

        # check batch
        m = error_message_batch(hyper_params["batch_size"])
        if len(m) > 0:
            messages.append(m)
    return messages


def check_create_model_params(params):
    messages = []

    # check dataset_id
    dataset_id = params.get("dataset_id", None)
    m = error_message_model_dataset_id(dataset_id)
    if len(m) > 0:
        messages.append(m)

    # check algorithm_id
    algorithm_id = params.get("algorithm_id", None)
    m = error_message_model_algorithm_id(algorithm_id)
    if len(m) > 0:
        messages.append(m)

    # check hyper parameters
    hyper_parameters = params.get("hyper_parameters", None)
    m = error_message_model_hyper_params(hyper_parameters)
    if len(m) > 0:
        messages.extend(m)

    # raise error
    if len(messages) > 0:
        raise InvalidRequestParamError("\n".join(messages))


def check_upadte_model_params(params):
    # nothing to check in v2.2.
    pass


def check_export_format(format):
    if format not in ["csv"]:
        raise InvalidRequestParamError("format {} is unavailable.".format(format))


def check_dir_exists(dirname):
    if not dirname.exists():
        raise DirectoryNotFound("Directory {} is not found.".format(dirname))


def check_model_deployed(model):
    deployed_model = storage.fetch_deployed_model(model["task_id"])
    if deployed_model is not None and model["id"] == deployed_model["id"]:
        return True
    return False


# To use dataset list, because dataset detail Information is not shown in dataset list.
def dataset_to_light_dict(dataset):
    return {
        'id': dataset["id"],
        'name': dataset["name"],
        'description': dataset["description"],
        'task_id': dataset["task_id"],
        'ratio': dataset["ratio"],
        'class_map': [],
        # 'class_map': dataset["class_map"],
        'class_info': dataset["class_info"],
        'train_data': {},   # TODO:元のにはなかった
        'valid_data': {},
        # 'valid_data': dataset["valid_data"],
        'test_dataset_id': dataset["test_dataset_id"]
    }


def dataset_to_dict(dataset):
    return {
        'id': dataset["id"],
        'name': dataset["name"],
        'description': dataset["description"],
        'task_id': dataset["task_id"],
        'ratio': dataset["ratio"],
        'class_map': dataset["class_map"],
        'class_info': dataset["class_info"],
        'train_data': dataset["train_data"],
        'valid_data': dataset["valid_data"],
        'test_dataset_id': dataset["test_dataset_id"],
    }


# To use model list, model detail information is not shown in model list.
def model_to_light_dict(model):
    return {
        "id": model["id"],
        "task_id": model["task_id"],
        "dataset_id": model["dataset_id"],
        "algorithm_id": model["algorithm_id"],
        "hyper_parameters": model["hyper_parameters"],
        "state": model["state"],
        "running_state": model["running_state"],
        "train_loss_list": [],
        # "train_loss_list": model["train_loss_list"],
        "valid_loss_list": [],
        # "valid_loss_list": model["valid_loss_list"],
        "best_epoch_valid_result": {},  # modify only evaluation value return
        # "best_epoch_valid_result": model["best_epoch_valid_result"],
        "total_epoch": model["total_epoch"],
        "nth_epoch": model["nth_epoch"],
        "total_batch": model["total_batch"],
        "nth_batch": model["nth_batch"],
        "last_batch_loss": model["last_batch_loss"],
        "last_prediction_result": {},
        "created": model["created"],
        "updated": model["updated"],
        "deployed": check_model_deployed(model)
    }


def model_to_dict(model):
    return {
        "id": model["id"],
        "task_id": model["task_id"],
        "dataset_id": model["dataset_id"],
        "algorithm_id": model["algorithm_id"],
        "hyper_parameters": model["hyper_parameters"],
        "state": model["state"],
        "running_state": model["running_state"],
        "train_loss_list": model["train_loss_list"],
        "valid_loss_list": model["valid_loss_list"],
        "best_epoch_valid_result": model["best_epoch_valid_result"],
        "total_epoch": model["total_epoch"],
        "nth_epoch": model["nth_epoch"],
        "total_batch": model["total_batch"],
        "nth_batch": model["nth_batch"],
        "last_batch_loss": model["last_batch_loss"],
        "last_prediction_result": model["last_prediction_result"],
        "created": model["created"],
        "updated": model["updated"],
        "deployed": check_model_deployed(model)
    }


def calc_class_ratio(train_num, valid_num):
    return ((train_num + valid_num) / np.sum(train_num + valid_num))


def calc_train_ratio(train_num, valid_num):
    return (train_num / (train_num + valid_num))


def calc_valid_ratio(train_num, valid_num):
    return (valid_num / (train_num + valid_num))


def class_info_to_dict(class_map, train_num, valid_num, test_ratio, train_img, valid_img):
    return {
        "class_map": class_map,
        "class_ratio": ndarray_to_list(calc_class_ratio(train_num, valid_num)),
        "train_ratio": ndarray_to_list(calc_train_ratio(train_num, valid_num)),
        "valid_ratio": ndarray_to_list(calc_valid_ratio(train_num, valid_num)),
        "test_ratio": test_ratio,
        "train_img_num": len(train_img),
        "valid_img_num": len(valid_img),
        "test_img_num": 1,
    }


def ndarray_to_list(data):
    return data.tolist()


def split_by_ratio(data, perm, ratio, length):
    return np.split(np.array([data[index] for index in perm]), [int(ratio * length)])


def get_tag_list(target):
    ret = []
    for i in range(len(target)):
        for j in range(len(target[i])):
            ret.append(target[i][j].get('class'))
    return ret


def parse_classification_target(file_names, img_dir):
    classification_label_dir = DATASET_LABEL_CLASSIFICATION_DIR
    class_labe_path = classification_label_dir / "target.txt"
    check_dir_exists(classification_label_dir)

    target, class_map = parse_txt_classification(str(class_labe_path))
    target_file_list = list(target.keys())

    file_names = [p for p in file_names
                  if (img_dir / p).is_file() and (p.name in target_file_list)]

    parsed_target = [target[name.name] for name in file_names]
    return parsed_target, class_map


def parse_detection_target(file_names, img_dir):
    detection_label_dir = DATASET_LABEL_DETECTION_DIR
    check_dir_exists(detection_label_dir)
    file_names = [p for p in file_names
                  if (img_dir / p).is_file() and ((detection_label_dir / p.name).with_suffix(".xml")).is_file()]

    xml_files = [str(detection_label_dir / name.with_suffix('.xml')) for name in file_names]
    parsed_target, class_map = parse_xml_detection(xml_files, num_thread=8)
    return parsed_target, class_map


def parse_segmentation_target(file_names, img_dir):
    segmentation_label_dir = DATASET_LABEL_SEGMENTATION_DIR
    check_dir_exists(segmentation_label_dir)
    file_names = [p for p in file_names if (img_dir / p).is_file() and
                  any([((segmentation_label_dir / p.name).with_suffix(suf)).is_file()
                       for suf in [".jpg", ".png"]])]
    parsed_target = [str(segmentation_label_dir / name.with_suffix(".png"))
                     for name in file_names]
    class_map = parse_classmap_file(str(segmentation_label_dir / "class_map.txt"))
    return parsed_target, class_map


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


# deprecated
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


def error_handler(func):
    def wrapped(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
            return ret
        except (InvalidRequestParamError) as e:
            logging_error(e)
            return create_error_response(e, status=400)
        except (TaskNotFoundError, DatasetNotFoundError, ModelNotFoundError, WeightNotFoundError) as e:
            logging_error(e)
            return create_error_response(e, status=404)
        except SQLAlchemyError as e:
            e = ServiceUnavailableError("DB tempolary unavailable.")
            logging_error(e)
            return create_error_response(e, status=503)
        except (DirectoryNotFound, Exception) as e:
            logging_error(e)
            return create_error_response(e, status=500)
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


@error(403)
def error403(error):
    e = ForbiddenError("Permission denied.")
    logging_error(e)
    return create_error_response(e, status=403)


@error(404)
def error404(error):
    e = NotFoundError("Endpoint not found.")
    logging_error(e)
    return create_error_response(e, status=404)


@error(405)
def error405(error):
    e = MethodNotAllowedError("Method not allowed.")
    logging_error(e)
    return create_error_response(e, status=405)


@route("/datasrc/<folder_name:path>/<file_name:path>")
def datasrc(folder_name, file_name):
    file_dir = os.path.join('datasrc', folder_name)
    return static_file(file_name, root=file_dir, mimetype='image/*')


@route("/api/renom_img/v2/model/<model_id:int>/export/", method="GET")
def export_csv(model_id):
    try:
        model = storage.fetch_model(model_id)
        prediction = model["last_prediction_result"]
        task_id = model["task_id"]
        print(task_id)
        ret = []
        if task_id == Task.CLASSIFICATION.value:
            img_path = prediction["img"]
            sizes = prediction["size"]
            prediction = prediction["prediction"]
            for img, size, pred in zip(img_path, sizes, prediction):
                ret.append({
                    'path': img,
                    'size': size,
                    'predictions': pred["class"]
                })

        elif task_id == Task.DETECTION.value:
            img_path = prediction["img"]
            sizes = prediction["size"]
            prediction = prediction["prediction"]
            for img, size, pred in zip(img_path, sizes, prediction):
                ret.append({
                    'path': img,
                    'size': size,
                    'predictions': pred
                })

        elif task_id == Task.SEGMENTATION.value:
            img_path = prediction["img"]
            sizes = prediction["size"]
            prediction = prediction["prediction"]
            for img, size, pred in zip(img_path, sizes, prediction):
                ret.append({
                    'path': img,
                    'size': size,
                    'predictions': pred
                })
        else:
            raise Exception("Not supported task id.")

        df = pd.DataFrame.from_dict(json_normalize(ret), orient='columns')
        df.to_csv('prediction.csv')
        return static_file("prediction.csv", root='.', download=True)

    except Exception as e:
        release_mem_pool()
        traceback.print_exc()
        body = json.dumps({"error_msg": "{}: {}".format(type(e).__name__, str(e))})
        ret = create_response(body, 500)
        return ret


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
    assert len(dataset_name) <= DATASET_NAME_MAX_LENGTH, \
        "Dataset name is too long. Please set the name length <= {}".format(DATASET_NAME_MAX_LENGTH)
    assert len(description) <= DATASET_DESCRIPTION_MAX_LENGTH, \
        "Dataset description is too long. Please set the description length <= {}".format(
            DATASET_DESCRIPTION_MAX_LENGTH)

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

    if task_id == Task.CLASSIFICATION.value:
        # Load data for Classification
        # Checks
        # 1. The class name file existence and format.

        classification_label_dir = DATASET_LABEL_CLASSIFICATION_DIR
        class_labe_path = classification_label_dir / "target.txt"

        assert classification_label_dir.exists(), \
            "target.txt was not found in the directory {}".format(str(classification_label_dir))

        target, class_map = parse_txt_classification(str(class_labe_path))
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

        train_tag_num, _ = np.histogram(train_target, bins=list(range(len(class_map) + 1)))
        valid_tag_num, _ = np.histogram(valid_target, bins=list(range(len(class_map) + 1)))

    elif task_id == Task.DETECTION.value:
        train_tag_list = []
        valid_tag_list = []

        for i in range(len(train_target)):
            for j in range(len(train_target[i])):
                train_tag_list.append(train_target[i][j].get('class'))

        for i in range(len(valid_target)):
            for j in range(len(valid_target[i])):
                valid_tag_list.append(valid_target[i][j].get('class'))

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


### New API ###
@route("/api/renom_img/v2/api/<task_name>/datasets", method="GET")
@error_handler
def get_datasets(task_name):
    """
    get datasets
    """
    task_id = get_task_id_by_name(task_name)
    datasets = storage.fetch_datasets_of_task(task_id)
    ret = {"datasets": [dataset_to_light_dict(d) for d in datasets]}
    return create_response(ret, status=200)

    # TODO：旧書式の下記の書き方だと正常にデータ取得可能。色々変わっている模様。
    # datasets = storage.fetch_datasets_of_task(task_id)
    # return {
    #     "datasets": [
    #         {
    #             'id': d["id"],
    #             'name': d["name"],
    #             'class_map': d["class_map"],
    #             'task_id': d["task_id"],
    #             'valid_data': d["valid_data"],
    #             'ratio': d["ratio"],
    #             'description': d["description"],
    #             'test_dataset_id': d["test_dataset_id"],
    #             'class_info': d["class_info"],
    #         }
    #         for d in datasets
    #     ]
    # }


@route("/api/renom_img/v2/api/<task_name>/datasets", method="POST")
@error_handler
def create_dataset(task_name):
    """
    create detection dataset
    """
    task_id = get_task_id_by_name(task_name)
    req_params = request.json
    check_create_dataset_params(req_params)

    # Receive params here.
    dataset_name = str(urllib.parse.unquote(req_params["name"], encoding='utf-8'))
    description = str(urllib.parse.unquote(req_params["description"], encoding='utf-8'))
    ratio = float(req_params["ratio"])

    try:
        test_dataset_id = int(req_params["test_dataset_id"])
    except KeyError:
        test_dataset_id = -1

    # TODO: Load root directory from configuration file or env params.
    # TODO: Make other storage available. e.g. S3.
    root = pathlib.Path('datasrc')
    img_dir = root / 'img'
    label_dir = root / 'label'
    check_dir_exists(img_dir)
    check_dir_exists(label_dir)

    file_names = [name.relative_to(img_dir) for name in img_dir.iterdir()
                  if name.is_file()]

    # ckeck test dataset exists
    if test_dataset_id > 0:
        test_dataset = storage.fetch_test_dataset(test_dataset_id)
        test_dataset_files = set([pathlib.Path(test_path).relative_to(img_dir)
                            for test_path in test_dataset['data']['img']])
        # Remove test files.
        file_names = file_names - test_dataset_files

    img_files = [str(img_dir / name) for name in file_names]

    # parse label data
    # TODO: create parser
    if task_id == Task.CLASSIFICATION.value:
        parsed_target, class_map = parse_classification_target(file_names, img_dir)
    elif task_id == Task.DETECTION.value:
        parsed_target, class_map = parse_detection_target(file_names, img_dir)
    elif task_id == Task.SEGMENTATION.value:
        parsed_target, class_map = parse_segmentation_target(file_names, img_dir)

    # Split into train and valid.
    n_imgs = len(file_names)
    perm = np.random.permutation(n_imgs)

    train_img, valid_img = split_by_ratio(img_files, perm, ratio, n_imgs)
    train_img = ndarray_to_list(train_img)
    valid_img = ndarray_to_list(valid_img)
    valid_img_size = [list(Image.open(i).size) for i in valid_img]

    train_target, valid_target = split_by_ratio(parsed_target, perm, ratio, n_imgs)
    train_target = ndarray_to_list(train_target)
    valid_target = ndarray_to_list(valid_target)

    test_ratio = []

    # count tag list
    if task_id == Task.CLASSIFICATION.value:
        train_tag_num, _ = np.histogram(train_target, bins=list(range(len(class_map) + 1)))
        valid_tag_num, _ = np.histogram(valid_target, bins=list(range(len(class_map) + 1)))
    elif task_id == Task.DETECTION.value:
        train_tag_list = get_tag_list(train_target)
        valid_tag_list = get_tag_list(valid_target)
        train_tag_num, _ = np.histogram(train_tag_list, bins=list(range(len(class_map) + 1)))
        valid_tag_num, _ = np.histogram(valid_tag_list, bins=list(range(len(class_map) + 1)))
    elif task_id == Task.SEGMENTATION.value:
        train_tag_num = parse_image_segmentation(train_target, len(class_map), 8)
        valid_tag_num = parse_image_segmentation(valid_target, len(class_map), 8)

    class_info = class_info_to_dict(class_map, train_tag_num, valid_tag_num, test_ratio, train_img, valid_img)

    train_data = {
        'img': train_img,
        'target': train_target
    }

    valid_data = {
        'img': valid_img,
        'target': valid_target,
        'size': valid_img_size,
    }

    # Registering in DB instead of temporary registration with global variable
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

    ret = {
        "dataset": {
            'id': dataset_id,
            'name': dataset_name,
            'description': description,
            'task_id': task_id,
            'ratio': ratio,
            'class_map': class_map,
            'class_info': class_info,
            'train_data': train_data,
            'valid_data': valid_data,
            'test_dataset_id': test_dataset_id
        }
    }
    response = create_response(ret, status=201)
    location = "/api/renom_img/v2/api/detection/datasets/{}".format(dataset_id)
    response.set_header('Location', location)
    return response


@route("/api/renom_img/v2/api/<task_name>/datasets/<dataset_id:int>", method="GET")
@error_handler
def get_dataset(task_name, dataset_id):
    """
    get dataset
    """
    task_exists(task_name)
    d = storage.fetch_dataset(dataset_id)
    check_dataset_exists(d, dataset_id)
    ret = {"dataset": dataset_to_dict(d)}
    return create_response(ret, status=200)


@route("/api/renom_img/v2/api/<task_name>/datasets/<dataset_id:int>", method="PUT")
@error_handler
def update_dataset(task_name, dataset_id):
    """
    update dataset
    """
    # TODO: Add dataset registration status columns in dataset table.
    # TODO: Register temporaly registed dataset.
    # task_exists(task_name)
    # return create_response({}, status=204)
    return "Update dataset is not available now."


@route("/api/renom_img/v2/api/<task_name>/datasets/<dataset_id:int>", method="DELETE")
@error_handler
def delete_dataset(task_name, dataset_id):
    """
    delete dataset
    """
    task_exists(task_name)
    d = storage.fetch_dataset(dataset_id)
    check_dataset_exists(d, dataset_id)
    storage.remove_dataset(dataset_id)
    return create_response({}, status=204)


@route("/api/renom_img/v2/api/<task_name>/models", method="GET")
@error_handler
def get_models(task_name):
    """
    get models
    """
    task_id = get_task_id_by_name(task_name)
    req_params = request.params
    state = req_params.state

    if state == "running":
        models = storage.fetch_running_models(task_id)
    elif state == "deployed":
        models = storage.fetch_deployed_model(task_id)
    else:
        models = storage.fetch_models_of_task(task_id)

    ret = {'models': [model_to_light_dict(m) for m in models]}
    return create_response(ret, status=200)

# 旧ソース
# def models_load_of_task(task_id):
#     models = storage.fetch_models_of_task(task_id)
#     # Remove best_valid_changed because it is very large.
#     models = [
#         {k: v if k not in [
#             "best_epoch_valid_result",
#             "last_prediction_result"] else {} for k, v in m.items()}
#         for m in models
#     ]
#     return {'model_list': models}



@route("/api/renom_img/v2/api/<task_name>/models", method="POST")
@error_handler
def create_model(task_name):
    # TODO: print("***server:create_model START")
    """
    create model
    """
    task_id = get_task_id_by_name(task_name)
    # TODO: print("***task_id:", task_id)
    # TODO: print("***request:", request)
    # TODO: print("***request.params:", request.params)
    # TODO: req_params = request.params

    # TODO: print("***algorithm_id:", json.loads(req_params.algorithm_id))
    # TODO: print("***dataset_id:", req_params.dataset_id)
    # TODO: print("***hyper_params:", json.loads(req_params.hyper_params))
    # TODO: print("***hyper_parameters:", json.loads(req_params.hyper_parameters))


    # req_json = request.params
    req_json = request.json
    print("***req_json:", req_json)
    check_create_model_params(req_json)

    hyper_params = req_json['hyper_parameters']
    algorithm_id = req_json['algorithm_id']
    dataset_id = req_json['dataset_id']

    new_id = storage.register_model(
        int(task_id), int(dataset_id), int(algorithm_id), hyper_params)

    ret = {
        "model": {
            "id": new_id,
            "task_id": task_id,
            "dataset_id": dataset_id,
            "algorithm_id": algorithm_id,
            "hyper_parameters": hyper_params
        }
    }
    response = create_response(ret, status=201)
    location = "/api/renom_img/v2/api/{}/models/{}".format(task_name, new_id)
    response.set_header('Location', location)
    return response


@route("/api/renom_img/v2/api/<task_name>/models/<model_id:int>", method="GET")
@error_handler
def get_model(task_name, model_id):
    """
    get model
    """
    task_exists(task_name)
    model = storage.fetch_model(model_id)
    check_model_exists(model, model_id)
    ret = {'model': model_to_dict(model)}
    return create_response(ret, status=200)


@route("/api/renom_img/v2/api/<task_name>/models/<model_id:int>", method="PUT")
@error_handler
def update_model(task_name, model_id):
    """
    change model's deploy status.
    """
    task_id = get_task_id_by_name(task_name)
    req_params = request.json

    check_upadte_model_params(req_params)

    model = storage.fetch_model(model_id)
    check_model_exists(model, model_id)

    # if deploy value exists
    deploy = req_params.pop("deploy", False)
    if deploy:
        check_model_running(model_id)
        storage.deploy_model(model_id)
    else:
        storage.undeploy_model(task_id)

    # update deploy status in v2.2.
    # storage.update_model(model_id, **req_params)
    return create_response({}, status=204)


@route("/api/renom_img/v2/api/<task_name>/models/<model_id:int>", method="DELETE")
@error_handler
def delete_model(task_name, model_id):
    """
    delete model
    """
    task_exists(task_name)
    model = storage.fetch_model(model_id)
    check_model_exists(model, model_id)

    threads = TrainThread.jobs
    active_train_thread = threads.get(id, None)
    if active_train_thread is not None:
        active_train_thread.stop()
        active_train_thread.future.result()
    storage.remove_model(model_id)
    return create_response({}, status=204)


@route("/api/renom_img/v2/api/<task_name>/models/<model_id:int>/weight", method="GET")
@error_handler
def download_model_weight(task_name, model_id):
    """
    download model weight file
    """
    task_exists(task_name)
    model = storage.fetch_model(model_id)
    check_model_exists(model, model_id)

    file_name = model['best_epoch_weight']
    check_weight_exists(file_name, model_id)

    download_filename = 'model{}_weight.h5'.format(model_id)
    return static_file(file_name, root=".", download=download_filename)


@route("/api/renom_img/v2/api/<task_name>/train", method="POST")
@error_handler
def run_train(task_name):
    """
    run train
    """
    task_exists(task_name)
    req_params = request.json
    model_id = req_params.get("model_id", None)
    model = storage.fetch_model(model_id)
    check_model_exists(model, model_id)

    # TODO: Confirm if the model is already trained.
    thread = TrainThread(model_id)
    th = executor.submit(thread)
    thread.set_future(th)

    # TODO: set train_id to thread
    # train_id = 1
    # response = create_response({"train": {"train_id": train_id}}, status=201)
    # location = "/api/renom_img/v2/api/detection/train/{}".format(train_id)
    # response.set_header('Location', location)
    response = create_response({"status": "success"}, status=201)
    return response


# TODO: getting train status from train_id.
# @route("/api/renom_img/v2/api/<task_name>/train/<train_id:int>", method="GET")
@route("/api/renom_img/v2/api/<task_name>/train", method="GET")
@error_handler
def get_train_status(task_name):
    """
    get train status
    """
    task_exists(task_name)
    # get query params
    req_params = request.params
    model_id = req_params.model_id

    threads = TrainThread.jobs
    active_train_thread = threads.get(model_id, None)
    if active_train_thread is None:
        saved_model = storage.fetch_model(model_id)
        check_model_exists(saved_model, model_id)

        # If the state == STOPPED, client will never throw request.
        if saved_model["state"] != State.STOPPED.value:
            storage.update_model(model_id, state=State.STOPPED.value,
                                 running_state=RunningState.STOPPING.value)
            saved_model = storage.fetch_model(model_id)

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


# TODO: Stop train from train_id.
# @route("/api/renom_img/v2/api/detection/train/<train_id:int>", method="DELETE")
@route("/api/renom_img/v2/api/<task_name>/train", method="DELETE")
@error_handler
def stop_train(task_name):
    """
    stop train
    """
    task_exists(task_name)
    req_params = request.json
    model_id = req_params["model_id"]

    saved_model = storage.fetch_model(model_id)
    check_model_exists(saved_model, model_id)

    thread = TrainThread.jobs.get(model_id, None)
    if thread is not None:
        thread.stop()
    return create_response({}, status=204)


@route("/api/renom_img/v2/api/<task_name>/prediction", method="GET")
@error_handler
def get_prediction_status(task_name):
    """
    get prediction status
    """
    task_exists(task_name)
    # get query params
    req_params = request.params
    model_id = req_params.model_id

    saved_model = storage.fetch_model(model_id)
    check_model_exists(saved_model, model_id)

    threads = PredictionThread.jobs
    active_prediction_thread = threads.get(model_id, None)
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


@route("/api/renom_img/v2/api/<task_name>/prediction", method="POST")
@error_handler
def run_prediction(task_name):
    """
    run prediction
    """
    task_exists(task_name)
    req_params = request.json
    model_id = req_params.get("model_id", None)
    saved_model = storage.fetch_model(model_id)
    check_model_exists(saved_model, model_id)

    file_name = saved_model['best_epoch_weight']
    check_weight_exists(file_name, model_id)

    thread = PredictionThread(model_id)
    th = executor.submit(thread)
    thread.set_future(th)

    # prediction_id = 1
    # response = create_response({"prediction": {"prediction_id": prediction_id}}, status=201)
    # location = "/api/renom_img/v2/api/detection/prediction/{}".format(prediction_id)
    # response.set_header('Location', location)
    response = create_response({"status": "success"}, status=201)
    return response


@route("/api/renom_img/v2/api/<task_name>/prediction/result", method="GET")
@error_handler
def get_prediction_result(task_name):
    """
    get prediction result
    format
    """
    task_id = get_task_id_by_name(task_name)
    # get query params
    req_params = request.params
    model_id = req_params.model_id
    format = req_params.format
    check_export_format(format)

    filename = 'prediction.csv'

    model = storage.fetch_model(model_id)
    prediction = model["last_prediction_result"]
    print(prediction)

    # Shaping data by task & format.
    resolver = get_formatter_resolver(task_id)
    formatter = resolver.resolve(format)
    df = formatter.format(prediction)

    # Export shaped data.
    # It is better to create writer for increasing format.
    # writer = get_writer(format)
    # writer.write(df, filename)
    df.to_csv(filename)

    return static_file(filename, root='.', download=True)


def get_app():
    return default_app()


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
    wsgiapp = get_app()
    httpd = wsgi_server.Server(wsgiapp, host=args.host, port=int(args.port))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
