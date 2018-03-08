# coding: utf-8

import argparse
import os
import json
import pkg_resources
import base64
import threading
import time
import urllib
from bottle import HTTPResponse, default_app, route, static_file, request, error

import wsgi_server
from python.train_thread import TrainThread
from python.prediction_thread import PredictionThread
from python.weight_download_thread import WeightDownloadThread
from python.utils.storage import storage


STATE_FINISHED = 2
STATE_DELETED = 3


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_SET_DIR = os.path.join(DATASET_DIR, 'train_set')
VALID_SET_DIR = os.path.join(DATASET_DIR, 'valid_set')
PREDICTION_SET_DIR = os.path.join(DATASET_DIR, 'prediction_set')

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

if not os.path.exists(PREDICTION_SET_DIR):
    os.makedirs(PREDICTION_SET_DIR)
OUT_DIR = os.path.join(PREDICTION_SET_DIR, 'out')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
IMG_DIR = os.path.join(PREDICTION_SET_DIR, 'img')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

for path in [TRAIN_SET_DIR, VALID_SET_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)
    XML_DIR = os.path.join(path, 'label')
    if not os.path.exists(XML_DIR):
        os.makedirs(XML_DIR)
    IMG_DIR = os.path.join(path, 'img')
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)


def create_response(body):
    r = HTTPResponse(status=200, body=body)
    r.set_header('Content-Type', 'application/json')
    return r


def find_thread(thread_id):
    for th in threading.enumerate():
        if "thread_id" in dir(th) and thread_id == th.thread_id:
            return th


def get_train_thread_count():
    count = 0
    for th in threading.enumerate():
        if isinstance(th, TrainThread):
            count += 1
    return count


@route("/")
def index():
    return pkg_resources.resource_string(__name__, "index.html")


@route("/css/<file_name:path>")
def css(file_name):
    return static_file(file_name, root='css/', mimetype='text/css')


@route("/build/<file_name:path>")
def static(file_name):
    return static_file(file_name, root='build/', mimetype='application/javascript')


@error(404)
def error404(error):
    body = json.dumps({"error_msg": "Page Not Found"})
    ret = create_response(body)
    return ret


@route("/dataset/<folder_name:path>/<item:path>/<file_name:path>")
def dataset(folder_name, item, file_name):
    file_dir = os.path.join('dataset', folder_name, item)
    return static_file(file_name, root=file_dir, mimetype='image/*')


@route("/api/renom_img/v1/projects", method="GET")
def get_projects():
    try:
        kwargs = {}
        if request.params.fields != '':
            kwargs["fields"] = request.params.fields

        if request.params.order_by != '':
            kwargs["order_by"] = request.params.order_by

        data = storage.fetch_projects(**kwargs)
        body = json.dumps(data)

    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects", method="POST")
def create_projects():
    name = request.params.project_name
    comment = request.params.project_comment

    try:
        project_id = storage.register_project(name, comment)
        data = {
            "project_id": project_id
        }
        body = json.dumps(data)
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>", method="GET")
def get_project(project_id):
    try:
        kwargs = {}
        if request.params.fields != '':
            kwargs["fields"] = request.params.fields

        data = storage.fetch_project(project_id, **kwargs)
        body = json.dumps(data)
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>", method="POST")
def update_project(project_id):
    pass


@route("/api/renom_img/v1/projects/<project_id:int>/models", method="GET")
def get_models(project_id):
    try:
        kwargs = {}
        if request.params.fields != '':
            kwargs["fields"] = request.params.fields

        model_ids = []
        if request.params.running_model_ids != '':
            model_ids = list(map(int, request.params.running_model_ids.split(",")))

        last_epochs = []
        if request.params.last_epochs != '':
            last_epochs = list(map(int, request.params.last_epochs.split(",")))

        deploy_model_id = None
        if request.params.deploy_model_id != '':
            deploy_model_id = int(request.params.deploy_model_id)

        model_count = int(request.params.model_count)

        for j in range(60):
            project = storage.fetch_project(project_id, fields='deploy_model_id')
            data = storage.fetch_models(project_id, **kwargs)

            if deploy_model_id != project["deploy_model_id"]:
                # If deploy model changed
                body = json.dumps(data)
                ret = create_response(body)
                return ret

            if model_count < len(data):
                # If model created
                valid_results = data[list(data.keys())[-1]]["best_epoch_validation_result"]
                if "bbox_path_list" in valid_results:
                    body = json.dumps(data)
                    ret = create_response(body)
                    return ret

            elif model_count > len(data):
                # If model deleted
                body = json.dumps(data)
                ret = create_response(body)
                return ret

            elif model_count == len(data):
                for i, v in enumerate(model_ids):
                    thread_id = "{}_{}".format(project_id, model_ids[i])
                    th = find_thread(thread_id)

                    if th is not None:
                        # If thread status updated, return response.
                        if last_epochs[i] != th.last_epoch:
                            body = json.dumps(data)
                            ret = create_response(body)
                            return ret
                    else:
                        # If running thread stopped, return response.
                        body = json.dumps(data)
                        ret = create_response(body)
                        return ret

            time.sleep(1)

    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/dataset_info", method="GET")
def get_dataset_info_v0():
    try:
        data = storage.fetch_dataset_v0()
        body = json.dumps(data)
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models", method="POST")
def create_model(project_id):
    # check training count
    if get_train_thread_count() >= 2:
        body = json.dumps({"error_msg": "You can create only 2 training thread."})
        ret = create_response(body)
        return ret

    try:
        model_id = storage.register_model(
            project_id=project_id,
            hyper_parameters=json.loads(request.params.hyper_parameters),
            algorithm=request.params.algorithm,
            algorithm_params=json.loads(request.params.algorithm_params))

        data = {"model_id": model_id}
        body = json.dumps(data)

    except Exception as e:
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
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>", method="DELETE")
def delete_model(project_id, model_id):
    try:
        thread_id = "{}_{}".format(project_id, model_id)

        # 学習中のスレッドを停止する
        th = find_thread(thread_id)
        if th is not None:
            th.stop()
        storage.update_model_state(model_id, STATE_DELETED)

    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/deploy", method="GET")
def deploy_model(project_id, model_id):
    try:
        storage.update_project_deploy(project_id, model_id)
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/undeploy", method="GET")
def undeploy_model(project_id, model_id):
    try:
        storage.update_project_deploy(project_id, None)
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/run", method="GET")
def run_model(project_id, model_id):
    try:
        # 学習データ読み込み
        fields = 'hyper_parameters,algorithm,algorithm_params'
        data = storage.fetch_model(project_id, model_id, fields=fields)

        # 学習を実行するスレッドを立てる
        thread_id = "{}_{}".format(project_id, model_id)
        th = TrainThread(thread_id, project_id, model_id,
                         data["hyper_parameters"],
                         data['algorithm'], data['algorithm_params'])
        th.start()
        th.join()
        if th.error_msg is not None:
            storage.update_model_state(model_id, STATE_FINISHED)
            body = json.dumps({"error_msg": th.error_msg})
            ret = create_response(body)
            return ret
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/running_info", method="GET")
def get_running_model_info(project_id, model_id):
    try:
        last_batch = int(request.params.last_batch)
        running_state = int(request.params.running_state)

        thread_id = "{}_{}".format(project_id, model_id)
        th = find_thread(thread_id)

        for i in range(60):
            # If thread status updated, return response.
            if th is not None:
                updated = (last_batch != th.last_batch) or (running_state != th.running_state)
                if updated is True:
                    body = json.dumps({
                        "last_batch": th.last_batch,
                        "total_batch": th.total_batch,
                        "last_train_loss": th.last_train_loss,
                        "running_state": th.running_state,
                    })
                    ret = create_response(body)
                    return ret
            time.sleep(1)
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/stop", method="GET")
def stop_model(project_id, model_id):
    try:
        # 学習中のスレッドを停止する
        thread_id = "{}_{}".format(project_id, model_id)

        th = find_thread(thread_id)
        th.running_state = 4
        if th is not None:
            th.stop()

    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/run_prediction", method="GET")
def run_prediction(project_id, model_id):
    # 学習データ読み込み
    try:
        fields = 'hyper_parameters,algorithm,algorithm_params,best_epoch_weight'
        data = storage.fetch_model(project_id, model_id, fields=fields)

        # weightのh5ファイルのパスを取得して予測する
        th = PredictionThread(data["hyper_parameters"], data["algorithm"], data["algorithm_params"], data["best_epoch_weight"])
        th.start()
        th.join()

        data = {
            "predict_results": th.predict_results,
            "csv": th.csv_filename,
        }
        body = json.dumps(data)
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/export_csv/<file_name:path>", method="GET")
def export_csv(project_id, model_id, file_name):
    return static_file(file_name, root="./.storage/csv", download=True)


@route("/api/renom_img/v1/original_img", method="POST")
def get_original_img():
    try:
        file_path = request.params.root_dir

        with open(file_path, "rb") as image_reader:
            encoded_img = base64.b64encode(image_reader.read())
            data = encoded_img.decode('utf8')
        body = json.dumps(data)

    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/weights/yolo", method="GET")
def check_weight_exists():
    try:
        if not os.path.exists("yolo.h5"):
            print("Weight parameters will be downloaded.")

            url = "http://docs.renom.jp/downloads/weights/yolo.h5"
            th = WeightDownloadThread("yolo", url, "yolo.h5")
            th.start()
        else:
            print("exists weight")
            body = json.dumps({"weight_exist": 1})
            ret = create_response(body)
            return ret

    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/weights/yolo/progress/<progress_num:int>", method="GET")
def check_weight_download_progress(progress_num):
    try:
        th = find_thread("yolo")
        for i in range(60):
            if th.percentage > 10 * progress_num:
                body = json.dumps({"progress": th.percentage})
                ret = create_response(body)
                return ret
            time.sleep(1)

    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/check_dir", method="GET")
def check_dataset_dir():
    try:
        files = os.listdir(os.path.join(TRAIN_SET_DIR, "label"))
        if len(files) == 0:
            raise Exception("File not found in train_set/label.")

        files = os.listdir(os.path.join(TRAIN_SET_DIR, "img"))
        if len(files) == 0:
            raise Exception("File not found in train_set/img.")

        files = os.listdir(os.path.join(VALID_SET_DIR, "label"))
        if len(files) == 0:
            raise Exception("File not found in valid_set/label.")

        files = os.listdir(os.path.join(VALID_SET_DIR, "img"))
        if len(files) == 0:
            raise Exception("File not found in valid_set/img.")
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='desc')
    parser.add_argument('--host', default='0.0.0.0', help='Server address')
    parser.add_argument('--port', default='8070', help='Server port')
    args = parser.parse_args()

    wsgiapp = default_app()
    httpd = wsgi_server.Server(wsgiapp, host=args.host, port=int(args.port))
    httpd.serve_forever()
