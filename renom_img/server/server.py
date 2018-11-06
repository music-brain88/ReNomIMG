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

# Constants
from renom_img.server import MAX_THREAD_NUM, DB_DIR_TRAINED_WEIGHT, GPU_NUM
from renom_img.server import DATASRC_IMG, DATASRC_LABEL, DATASRC_DIR, DATASRC_PREDICTION_OUT
from renom_img.server import STATE_FINISHED, STATE_RUNNING, STATE_DELETED, STATE_RESERVED
from renom_img.server import WEIGHT_EXISTS, WEIGHT_CHECKING, WEIGHT_DOWNLOADING

# Thread(Future object) is stored to thread_pool as pair of "thread_id:[future, thread_obj]".
train_thread_pool = {}
prediction_thread_pool = {}

def get_train_thread_count():
    return len([th for th in train_thread_pool.values() if th[0].running()])

def create_response(body):
    r = HTTPResponse(status=200, body=body)
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

def json_handler(func):
    def wrapped(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
            if ret is None:
              ret = {}
            assert isinstance(ret, dict),\
              "The returned object of the API '{}' is not a dictionary.".format(func.__name__)
            body = json.dumps(ret, ignore_nan=True)
            return create_response(body)
        except Exception as e:
            release_mem_pool()
            traceback.print_exc()
            body = json.dumps({"error_msg": str(e)})
            ret = create_response(body)
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
    print(error)
    body = json.dumps({"error_msg": "Page Not Found"})
    ret = create_response(body)
    return ret


@route("/datasrc/<folder_name:path>/<file_name:path>")
def datasrc(folder_name, file_name):
    file_dir = os.path.join('datasrc', folder_name)
    return static_file(file_name, root=file_dir, mimetype='image/*')


####### WEB APIs
@route("/api/renom_img/v2/model/create", method="POST")
@json_handler
def model_create():
    req_params = request.params
    hyper_params = json.loads(json.dumps(req_params.hyper_params))
    parents = json.loads(json.dumps(req_params.parents))
    dataset_id = req_params.dataset_id
    task = req_params.task
    model_id = 1
    return {"id": model_id}

@route("/api/renom_img/v2/model/load/<id:int>", method="GET")
@json_handler
def model_load(id):
    return {"id": id}

@route("/api/renom_img/v2/model/remove/<id:int>", method="GET")
@json_handler
def model_remove(id):
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
