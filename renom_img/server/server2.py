# coding: utf-8

import os
import json
import time
import base64
import threading
import pkg_resources
import argparse
import urllib
import mimetypes
import posixpath
import traceback
import pathlib
import random
import xmltodict
from concurrent.futures import ThreadPoolExecutor as Executor
from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
from bottle import HTTPResponse, default_app, route, static_file, request, error

from renom.cuda import release_mem_pool

from renom_img.server import wsgi_server
from renom_img.server.train_thread2 import TrainThread
from renom_img.server.utility.storage import storage

# Constants
from renom_img.server import MAX_THREAD_NUM
from renom_img.server import DATASRC_IMG, DATASRC_LABEL
from renom_img.server import STATE_FINISHED, STATE_RUNNING, STATE_DELETED, STATE_RESERVED

executor = Executor(max_workers=MAX_THREAD_NUM)

# Thread(Future object) is stored to thread_pool as pair of "thread_id:[future, thread_obj]".
thread_pool = {}


def get_train_thread_count():
    return len([th for th in thread_pool.values() if th[0].running()])


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


@route("/")
def index():
    return _get_resource('', 'index.html')


@route("/static/<file_name:re:.+>")
def static(file_name):
    return _get_resource('static', file_name)


@route("/css/<file_name:path>")
def css(file_name):
    return _get_resource('static/css/', file_name)


@error(404)
def error404(error):
    print(error)
    body = json.dumps({"error_msg": "Page Not Found"})
    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>", method="GET")
def get_project(project_id):
    try:
        kwargs = {}
        kwargs["fields"] = "project_id,project_name,project_comment,deploy_model_id"

        data = storage.fetch_project(project_id, **kwargs)
        body = json.dumps(data)

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/model/create", method="POST")
def create_model(project_id):
    print("CALL create_model")
    try:
        model_id = storage.register_model(
            project_id=project_id,
            dataset_def_id=json.loads(request.params.dataset_def_id),
            hyper_parameters=json.loads(request.params.hyper_parameters),
            algorithm=request.params.algorithm,
            algorithm_params=json.loads(request.params.algorithm_params))

        if get_train_thread_count() >= MAX_THREAD_NUM:
            storage.update_model_state(model_id, STATE_RESERVED)
        else:
            storage.update_model_state(model_id, STATE_RUNNING)

        data = {"model_id": model_id}
        body = json.dumps(data)
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/run", method="GET")
def run_model(project_id, model_id):
    """
    Create thread(Future object) and submit it to executor.
    The thread is stored to thread_pool as a pair of thread_id and thread.

    """
    try:
        fields = 'hyper_parameters,algorithm,algorithm_params,dataset_def_id'
        data = storage.fetch_model(project_id, model_id, fields=fields)
        rec = storage.fetch_dataset_def(data['dataset_def_id'])
        (id, name, ratio, train_imgs, valid_imgs, created, updated) = rec
        thread_id = "{}_{}".format(project_id, model_id)
        th = TrainThread(thread_id, project_id, model_id,
                         data["hyper_parameters"],
                         data['algorithm'], data['algorithm_params'],
                         train_imgs, valid_imgs)
        ft = executor.submit(th)
        thread_pool[thread_id] = [ft, th]

        try:
            # This will wait for end of thread.
            print("Return of thread", ft.result())
        except CancelledError as ce:
            # If the model is deleted or stopped,
            # program reaches here.
            pass
        model = storage.fetch_model(project_id, model_id, fields='state')
        if model['state'] != STATE_DELETED:
            storage.update_model_state(model_id, STATE_FINISHED)
        release_mem_pool()
        if th.error_msg is not None:
            body = json.dumps({"error_msg": th.error_msg})
            ret = create_response(body)
            return ret

    except Exception as e:
        release_mem_pool()
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/dataset_defs", method="GET")
def get_datasets():
    try:
        recs = storage.fetch_dataset_defs()
        ret = []
        for rec in recs:
            id, name, ratio, created, updated = rec
            created = created.isoformat()
            updated = updated.isoformat()
            ret.append(dict(id=id, name=name, ratio=ratio, created=created, updated=updated))
        return create_response(json.dumps({'dataset_defs': ret}))

    except Exception as e:
        print(e)
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/dataset_defs/", method="POST")
def create_dataset_def():
    try:
        datasrc = pathlib.Path(DATASRC_DIR)
        imgdirname = pathlib.Path("img")
        xmldirname = pathlib.Path("label")

        imgdir = (datasrc / imgdirname)
        xmldir = (datasrc / xmldirname)

        name = request.params.name
        ratio = float(request.params.ratio)

        # search image files
        imgs = (p.relative_to(imgdir) for p in imgdir.iterdir() if p.is_file())

        # remove images without label
        imgs = {img for img in imgs if (xmldir / img).with_suffix('.xml').is_file()}

        # split files into trains and validations
        n_imgs = len(imgs)

        trains = set(random.sample(imgs, int(ratio * n_imgs)))
        valids = imgs - trains

        # build filename of images and labels
        train_imgs = [str(img) for img in trains]
        valid_imgs = [str(img) for img in valids]

        # register dataset
        id = storage.register_dataset_def(name, ratio, train_imgs, valid_imgs)
        body = json.dumps({"id": id})
        ret = create_response(body)
        return ret

    except Exception as e:
        print(e)
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


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
