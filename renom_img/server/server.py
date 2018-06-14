# coding: utf-8

import argparse
import os
import json
import pkg_resources
import base64
import threading
import time
import urllib
import pkg_resources
import mimetypes
import posixpath
import traceback
import xmltodict
from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
from bottle import HTTPResponse, default_app, route, static_file, request, error
from sklearn.model_selection import train_test_split

# signal(SIGPIPE, SIG_DFL)
from renom.cuda import release_mem_pool
from renom_img.server import wsgi_server
from renom_img.server.train_thread import TrainThread, WEIGHT_DIR
from renom_img.server.prediction_thread import PredictionThread
from renom_img.server.weight_download_thread import WeightDownloadThread
from renom_img.server.utility.storage import storage
from renom_img.server.utility.console_funcs import divide_datasets

STATE_RUNNING = 1
STATE_FINISHED = 2
STATE_DELETED = 3
STATE_RESERVED = 4

BASE_DIR = os.path.abspath(os.getcwd())
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_SET_DIR = os.path.join(DATASET_DIR, 'train_set')
VALID_SET_DIR = os.path.join(DATASET_DIR, 'valid_set')
PREDICTION_SET_DIR = os.path.join(DATASET_DIR, 'prediction_set')

MAX_THREAD_NUMBER = 2

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

semaphore = threading.BoundedSemaphore(MAX_THREAD_NUMBER)


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
        traceback.print_exc()
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
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})

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


@route("/api/renom_img/v1/projects/<project_id:int>", method="POST")
def update_project(project_id):
    pass


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


@route("/api/renom_img/v1/projects/<project_id:int>/models/update", method="GET")
def update_models(project_id):
    try:
        model_count = int(request.params.model_count)

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
                # If model created
                valid_results = data[list(data.keys())[-1]]["best_epoch_validation_result"]
                if "bbox_path_list" in valid_results:
                    body = json.dumps({
                        "models": data,
                        "update_type": 0
                    })
                    ret = create_response(body)
                    return ret

            # elif model_count > len(data):
            #     body = json.dumps({
            #         "models": {},
            #         "update_type": 1
            #     })
            #     ret = create_response(body)
            #     return ret

            elif model_count == len(data) or model_count >len(data):
                # if running information change, return response.
                for i, v in enumerate(model_ids):
                    thread_id = "{}_{}".format(project_id, model_ids[i])
                    th = find_thread(thread_id)

                    if th is not None:
                        # If thread status updated, return response.
                        if last_epochs[i] != th.last_epoch or running_states[i] != th.running_state:
                            body = json.dumps({
                                "models": running_models,
                                "update_type": 2
                            })
                            ret = create_response(body)
                            return ret
    except Exception as e:
        time.sleep(1000)
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


@route("/api/renom_img/v1/dataset_info", method="GET")
def get_dataset_info_v0():
    try:
        data = storage.fetch_dataset_v0()
        body = json.dumps(data)
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models", method="POST")
def create_model(project_id):
    try:
        model_id = storage.register_model(
            project_id=project_id,
            hyper_parameters=json.loads(request.params.hyper_parameters),
            algorithm=request.params.algorithm,
            algorithm_params=json.loads(request.params.algorithm_params))
        # 学習中のモデルが2つ以上ある場合には、モデルの状態をReservedに変更する
        if get_train_thread_count() >= MAX_THREAD_NUMBER:
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


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>", method="DELETE")
def delete_model(project_id, model_id):
    try:
        thread_id = "{}_{}".format(project_id, model_id)

        # 学習中のスレッドを停止する
        th = find_thread(thread_id)
        if th is not None:
            th.stop()
            th.join()
        storage.update_model_state(model_id, STATE_DELETED)

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


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/cancel", method="DELETE")
def cancel_model(project_id, model_id):
    try:
        thread_id = "{}_{}".format(project_id, model_id)
        print('cancel', STATE_DELETED)
        storage.update_model_state(model_id, STATE_DELETED)
        # 学習中のスレッドを停止する
        th = find_thread(thread_id)
        if th is not None:
            th.stop()
        while True:
            print('stop flag', th.stop_event.is_set(), 'thread_id ', thread_id)
            time.sleep(1)

    except Exception as e:
        traceback.print_exec()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/progress", method="GET")
def progress_model(project_id, model_id):
    try:
        thread_id = "{}_{}".format(project_id, model_id)

        fields = "model_id,project_id,state,last_epoch,last_batch,total_batch,last_train_loss,running_state"
        model = storage.fetch_model(project_id, model_id, fields=fields)
        if not model:
            body = json.dumps(model)
            ret = create_response(body)
            time.sleep(1)
            return ret
        for j in range(60):
            th = find_thread(thread_id)
            if th is not None:
                # If thread status updated, return response.
                if model["last_batch"] != th.last_batch or model["running_state"] != th.running_state or model["last_epoch"] != th.last_epoch:
                    body = json.dumps(model)
                    ret = create_response(body)
                    return ret
                time.sleep(1)
            else:
                # If thread status updated, return response.
                body = json.dumps(model)
                ret = create_response(body)
                return ret
                time.sleep(1)

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/deploy", method="GET")
def deploy_model(project_id, model_id):
    try:
        storage.update_project_deploy(project_id, model_id)
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/undeploy", method="GET")
def undeploy_model(project_id, model_id):
    try:
        storage.update_project_deploy(project_id, None)
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/deployed_model", method="GET")
def pull_deployed_model(project_id):
    # This method will be called from python script.
    try:
        deployed_id = storage.fetch_deployed_model_id(project_id)[0]['deploy_model_id']
        ret = storage.fetch_model(project_id, deployed_id, "best_epoch_weight")
        file_name = ret['best_epoch_weight']
        path = pkg_resources.resource_filename(__name__,
                                               posixpath.join('.build', '..', '..', '.storage', 'weight'))
        return static_file(file_name, root=path, download='deployed_model.h5')
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/deployed_model_info", method="GET")
def get_deployed_model_info(project_id):
    # This method will be called from python script.
    try:
        deployed_id = storage.fetch_deployed_model_id(project_id)[0]['deploy_model_id']
        ret = storage.fetch_model(project_id, deployed_id,
                                  "algorithm,algorithm_params,hyper_parameters")
        body = json.dumps(ret)
        ret = create_response(body)
        return ret
    except Exception as e:
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/run", method="GET")
def run(project_id, model_id):
    try:
        # 学習データが存在するかチェック
        files = os.listdir(os.path.join(TRAIN_SET_DIR, "label"))
        if len(files) == 0:
            raise Exception(
                "Error: File not found in train_set/label. You can find hints for this error on 'http://www.renom.jp/'.")

        files = os.listdir(os.path.join(TRAIN_SET_DIR, "img"))
        if len(files) == 0:
            raise Exception(
                "Error: File not found in train_set/img. You can find hints for this error on 'http://www.renom.jp/'.")

        # バリデーションデータが存在するかチェック
        files = os.listdir(os.path.join(VALID_SET_DIR, "label"))
        if len(files) == 0:
            raise Exception(
                "Error: File not found in valid_set/label. You can find hints for this error on 'http://www.renom.jp/'.")

        files = os.listdir(os.path.join(VALID_SET_DIR, "img"))
        if len(files) == 0:
            raise Exception(
                "Error: File not found in valid_set/img. You can find hints for this error on 'http://www.renom.jp/'.")

        # 学習データ読み込み
        fields = 'hyper_parameters,algorithm,algorithm_params'
        data = storage.fetch_model(project_id, model_id, fields=fields)

        thread_id = "{}_{}".format(project_id, model_id)
        th = TrainThread(thread_id, project_id, model_id,
                         data["hyper_parameters"],
                         data['algorithm'], data['algorithm_params'], semaphore)
        th.start()
        th.join()
        # Following line should be implemented here. Not in train_thread.py
        if not th.stop_event.is_set():
            storage.update_model_state(model_id, STATE_FINISHED)
        release_mem_pool()
        if th.error_msg is not None:
            storage.update_model_state(model_id, STATE_FINISHED)
            body = json.dumps({"error_msg": th.error_msg})
            ret = create_response(body)
            return ret
    except Exception as e:
        release_mem_pool()
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/stop", method="GET")
def stop_model(project_id, model_id):
    try:
        # 学習中のスレッドを停止する
        thread_id = "{}_{}".format(project_id, model_id)

        th = find_thread(thread_id)
        if th is not None:
            storage.update_model_state(model_id, STATE_FINISHED)
            th.running_state = 4
            th.stop()

    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/run_prediction", method="GET")
def run_prediction(project_id, model_id):
    # 学習データ読み込み
    try:
        thread_id = "{}_{}".format(project_id, model_id)
        fields = 'hyper_parameters,algorithm,algorithm_params,best_epoch_weight'
        data = storage.fetch_model(project_id, model_id, fields=fields)

        # weightのh5ファイルのパスを取得して予測する
        th = PredictionThread(thread_id, data["hyper_parameters"], data["algorithm"],
                              data["algorithm_params"], data["best_epoch_weight"])
        th.start()
        th.join()

        if th.error_msg is not None:
            body = json.dumps({"error_msg": th.error_msg})
        else:
            data = {
                "predict_results": th.predict_results,
                "csv": th.csv_filename,
            }
            body = json.dumps(data)
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})

    ret = create_response(body)
    return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/prediction_info", method="GET")
def prediction_info(project_id, model_id):
    # 学習データ読み込み
    time.sleep(1)
    thread_id = "{}_{}".format(project_id, model_id)
    th = find_thread(thread_id)
    if th is not None:
        data = {
            "predict_total_batch": th.total_batch,
            "predict_last_batch": th.last_batch,
        }
        body = json.dumps(data)
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/projects/<project_id:int>/models/<model_id:int>/export_csv/<file_name:path>", method="GET")
def export_csv(project_id, model_id, file_name):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(BASE_DIR, "../.storage/csv")
    return static_file(file_name, root=csv_dir, download=True)


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
        traceback.print_exc()
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
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/datasets", method="GET")
def load_datasets():
    try:
        datasets = storage.fetch_datasets()
        body = json.dumps({"datasets": datasets})
        ret = create_response(body)
        return ret
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/datasets", method="POST")
def create_dataset():
    try:
        dataste_name = request.params.dataset_name
        train_size = float(request.params.train_size)
        type = int(request.params.type)
        row = storage.register_dataset(dataste_name, train_size, type)
        body = json.dumps({"row": row})
        ret = create_response(body)
        return ret
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


@route("/api/renom_img/v1/datasets/<dataset_id:int>", method="GET")
def load_dataset(dataset_id):
    try:
        dataset = storage.fetch_dataset(dataset_id)
        body = json.dumps({"dataset": dataset})
        ret = create_response(body)
        return ret
    except Exception as e:
        traceback.print_exc()
        body = json.dumps({"error_msg": e.args[0]})
        ret = create_response(body)
        return ret


def xml2json(xml_file, xml_attribs=True):
    with open(xml_file, "rb") as f:  # notice the "rb" mode
        d = xmltodict.parse(f, xml_attribs=xml_attribs)
        return json.dumps(d, indent=4)


@route("/api/renom_img/v1/datasets/<dataset_id:int>", method="POST")
def update_dataset(dataset_id):
    try:
        data = storage.fetch_dataset_train_size(dataset_id)
        img_dir = request.params.img_dir
        label_dir = request.params.label_dir

        imgs = os.listdir(img_dir)
        labels = os.listdir(label_dir)

        train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(
            imgs, labels,     train_size=float(data["train_size"]))

        # get class name info form xml
        class_names = []
        for l in labels:
            json_data = xml2json(os.path.join(label_dir, l))
            json_dict = json.loads(json_data)
            try:
                # object dataが1つだけの場合、dictになってしまうのでlistに変換する
                if isinstance(json_dict['annotation']['object'], dict):
                    temp = [json_dict['annotation']['object']]
                    json_dict['annotation']['object'] = temp
            except KeyError:
                json_dict['annotation']['object'] = ''

            objects = json_dict["annotation"]["object"]
            for o in objects:
                if o["name"] not in class_names:
                    class_names.append(o["name"])

        storage.update_dataset(dataset_id, len(train_imgs),
                               len(valid_imgs), len(class_names),
                               len(class_names), train_imgs,
                               valid_imgs, train_labels,
                               valid_labels, class_names,
                               img_dir, label_dir)

        body = json.dumps({})
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

    # Add divide function.
    help = 'Divide image and labels into train set and valid set. '
    help += 'Users can pass division ratio, with this argument. '
    help += 'If the ratio is not passed, default ratio(0.8) will be used. '

    subparsers = parser.add_subparsers()
    parser_divide = subparsers.add_parser('divide')

    help = 'Division ratio. If r=0.8, dataset will be divided into train(0.8):varid(0.2).'
    parser_divide.add_argument('r', type=float, default=0.8, help=help, nargs='?')
    parser_divide.set_defaults(divide_func=divide_datasets)

    args = parser.parse_args()
    if hasattr(args, 'divide_func'):
        args.divide_func(args.r)
    else:
        wsgiapp = default_app()
        httpd = wsgi_server.Server(wsgiapp, host=args.host, port=int(args.port))
        httpd.serve_forever()


if __name__ == "__main__":
    main()
