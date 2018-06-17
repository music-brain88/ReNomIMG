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

from renom_img.api.utility.load import parse_xml_detection

from renom_img.server import wsgi_server
from renom_img.server.train_thread2 import TrainThread
from renom_img.server.utility.storage import storage

# Constants
from renom_img.server import MAX_THREAD_NUM
from renom_img.server import DATASRC_IMG, DATASRC_LABEL, DATASRC_DIR
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

@route("/datasrc/<folder_name:path>/<file_name:path>")
def datasrc(folder_name, file_name):
    file_dir = os.path.join('datasrc', folder_name)
    return static_file(file_name, root=file_dir, mimetype='image/*')

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

@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>", method="GET")
def get_model(project_id, model_id):
    try:
        kwargs = {}
        if request.params.fields != '':
            kwargs["fields"] = request.params.fields

        data = storage.fetch_model(project_id, model_id, **kwargs)
        body = json.dumps(data)

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models", method="GET")
def get_models(project_id):
    # TODO: Cache validation img path on browser.
    try:
        data = storage.fetch_models(project_id)
        body = json.dumps(data)
        ret = create_response(body)
        return ret

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
        thread_id = "{}_{}".format(project_id, model_id)
        th = TrainThread(thread_id, project_id, model_id,
                         data['dataset_def_id'],
                         data["hyper_parameters"],
                         data['algorithm'], data['algorithm_params'])
        ft = executor.submit(th)
        thread_pool[thread_id] = [ft, th]

        try:
            # This will wait for end of thread.
            print("Return of thread", ft.result())
        except CancelledError as ce:
            # If the model is deleted or stopped,
            # program reaches here.
            traceback.print_exc()
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


@route("/api/renom_img/v1/projects/<project_id:int>/models/update", method="GET")
def update_models(project_id):
    try:
        model_count = int(request.params.model_count)
        # This gets running models only.
        running_models = storage.fetch_running_models(project_id)
        running_count = len(running_models)
        # set running model information for polling
        model_ids = []
        last_epochs = []
        last_batchs = []
        running_states = []
        for k in list(running_models.keys()):
            model_ids.append(running_models[k]["model_id"])
            last_epochs.append(running_models[k]["last_epoch"])
            last_batchs.append(running_models[k]["last_batch"])
            running_states.append(running_models[k]["running_state"])

        for j in range(300):
            time.sleep(1)
            data = storage.fetch_models(project_id)
            running_models = storage.fetch_running_models(project_id)
            if model_count < len(data):
                body = json.dumps({
                    "models": data,
                })
                ret = create_response(body)
                return ret
            else:
                raise Exception("Never reach here.")

    except Exception as e:
        time.sleep(1000)
        traceback.print_exc()


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/progress", method="GET")
def progress_model(project_id, model_id):
    try:
        req_last_batch = request.params.last_batch
        req_last_batch = int(req_last_batch) if req_last_batch else 0

        req_last_epoch = request.params.last_epoch
        req_last_epoch = int(req_last_epoch) if req_last_epoch else 0

        req_running_state = request.params.running_state
        req_running_state = int(req_running_state) if req_running_state else 0

        thread_id = "{}_{}".format(project_id, model_id)
        for j in range(60):
            time.sleep(1.5)
            th = thread_pool.get(thread_id, None)
            model_state = storage.fetch_model(project_id, model_id, "state")["state"]
            if th is not None:
                th = th[1]
                # If thread status updated, return response.
                if th.nth_batch != req_last_batch or \
                        th.running_state != req_running_state or \
                        th.nth_epoch != req_last_epoch:
                    body = json.dumps({
                        "total_batch": th.total_batch,
                        "last_batch": th.nth_batch,
                        "last_epoch": th.nth_epoch,
                        "batch_loss": th.last_batch_loss,
                        "running_state": th.running_state,
                        "state": model_state
                    })
                    ret = create_response(body)
                    return ret


    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/stop", method="GET")
def stop_model(project_id, model_id):
    try:
        thread_id = "{}_{}".format(project_id, model_id)

        th = thread_pool.get(thread_id, None)
        if th is not None:
            if not th[0].cancel():
                th[1].stop()
                th[0].result() # Same as join.
            storage.update_model_state(model_id, STATE_FINISHED)

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>", method="DELETE")
def delete_model(project_id, model_id):
    try:
        thread_id = "{}_{}".format(project_id, model_id)

        storage.update_model_state(model_id, STATE_DELETED)
        th = thread_pool[thread_id]
        if th is not None:
            if not th[0].cancel():
                th[1].stop()
                th[0].result()

        ret = storage.fetch_model(project_id, model_id, "best_epoch_weight")
        file_name = ret.get('best_epoch_weight', None)
        if file_name is not None:
            weight_path = os.path.join(WEIGHT_DIR, file_name)
            if os.path.exists(weight_path):
                os.remove(weight_path)

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret



@route("/api/renom_img/v1/projects/<project_id:int>/models/update/state", method="GET")
def update_models_state(project_id):
    try:
        models = storage.fetch_models(project_id)
        # set running model information for polling
        body = {}
        for k in list(models.keys()):
            model_id = models[k]["model_id"]
            running_state = models[k]["running_state"]
            state = models[k]['state']
            body[model_id] = {
                'running_state': running_state,
                'state': state
            }
        body = json.dumps(body)
        ret = create_response(body)
        return ret
    except Exception as e:
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
            id, name, ratio, valid_imgs, class_map, created, updated = rec
            valid_imgs = [os.path.join("datasrc/img/", path) for path in valid_imgs]
            ret.append(dict(id=id, name=name, ratio=ratio,
                valid_imgs=valid_imgs, class_map=class_map, created=created, updated=updated))
        return create_response(json.dumps({'dataset_defs': ret}))

    except Exception as e:
        traceback.print_exc()
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

        _, class_map = parse_xml_detection([str(path) for path in xmldir.iterdir()])
        class_map = [k for k, v in sorted(class_map.items(), key=lambda x:x[0])]

        # register dataset
        id = storage.register_dataset_def(name, ratio, train_imgs, valid_imgs, class_map)

        # Insert detailed informations
        train_num = len(train_imgs)
        valid_num = len(valid_imgs)

        body = json.dumps({"id": id})
        ret = create_response(body)
        return ret

    except Exception as e:
        traceback.print_exc()
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
