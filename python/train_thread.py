import os
import time
import threading
import numpy as np
from renom.cuda import set_cuda_active
from .model.yolo import YoloDarknet
from .utils.data_preparation import create_train_valid_dists
from .utils.storage import storage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.join(BASE_DIR, "../.storage/weight")
CSV_DIR = os.path.join(BASE_DIR, "../.storage/csv")
IMG_DIR = os.path.join(BASE_DIR, "../dataset/prediction_set/img")
XML_DIR = os.path.join(BASE_DIR, "../dataset/prediction_set/label")

STATE_RUNNING = 1
STATE_FINISHED = 2

TRAIN = 0
VALID = 1
PRED = 2
ERROR = -1

DEBUG = True


class TrainThread(threading.Thread):
    def __init__(self, thread_id, project_id, model_id, hyper_parameters,
                 algorithm, algorithm_params):
        super(TrainThread, self).__init__()
        self.stop_event = threading.Event()
        self.setDaemon(False)

        self.thread_id = thread_id
        self.project_id = project_id
        self.model_id = model_id
        self.algorithm = algorithm
        self.total_epoch = int(hyper_parameters['total_epoch'])
        self.batch_size = int(hyper_parameters['batch_size'])
        self.seed = int(hyper_parameters['seed'])
        self.img_size = (hyper_parameters['image_width'], hyper_parameters['image_height'])

        self.cell_h = int(algorithm_params['cells'])
        self.cell_v = int(algorithm_params['cells'])
        self.num_bbox = int(algorithm_params['bounding_box'])

        self.last_batch = 0
        self.total_batch = 0
        self.last_train_loss = None
        self.last_epoch = 0
        self.best_validation_loss = None
        self.running_state = 3

        self.model = None
        self.error_msg = None

    def get_iou_and_mAP(self, truth, predict_list):
        try:
            iou_list = []
            map_true_count = 0
            map_count = 0

            for i in range(len(truth)):
                label = truth[i].reshape(-1, 4 + self._class_num)
                pred = predict_list[i]

                for j in range(len(label)):
                    x, y, w, h = label[j, :4]
                    obj_class = np.argmax(label[j, 4:])

                    x1 = x - w / 2.
                    y1 = y - h / 2.
                    x2 = x + w / 2.
                    y2 = y + h / 2.

                    if x == y == w == h == 0:
                        break
                    map_count += 1
                    for k in range(len(pred)):
                        p_class = pred[k]['class']
                        if p_class != obj_class:
                            break
                        p_x = pred[k]['box'][0] * self.img_size[0]
                        p_y = pred[k]['box'][1] * self.img_size[1]
                        p_w = pred[k]['box'][2] * self.img_size[0]
                        p_h = pred[k]['box'][3] * self.img_size[1]
                        px1 = p_x - p_w / 2.
                        py1 = p_y - p_h / 2.
                        px2 = p_x + p_w / 2.
                        py2 = p_y + p_h / 2.

                        overlapped_dpx = min(px2, x2) - max(px1, x1)
                        overlapped_dpy = min(py2, y2) - max(py1, y1)
                        if overlapped_dpx <= 0 or overlapped_dpy <= 0:
                            break
                        intersection = overlapped_dpx * overlapped_dpy
                        union = p_w * p_h + w * h - intersection
                        iou = intersection / float(union)
                        iou_list.append(iou)
                        map_true_count += 1

            map = map_true_count / map_count
            return iou_list, map
        except Exception as e:
            self.error_msg = e.args[0]

    def run(self):
        try:
            set_cuda_active(True)
            print("run thread")
            storage.update_model_state(self.model_id, STATE_RUNNING)
            class_list, train_dist, valid_dist = create_train_valid_dists(self.img_size)
            storage.register_dataset_v0(len(train_dist), len(valid_dist), class_list)
            self.model = self.set_train_config(len(class_list))
            self.run_train(train_dist, valid_dist)
        except Exception as e:
            self.error_msg = e.args[0]

    def run_train(self, train_distributor, validation_distributor=None):
        try:
            # Prepare validation images for UI.
            valid_img = validation_distributor._data_table
            v_bbox_imgs = valid_img

            storage.update_model_validation_result(
                model_id=self.model_id,
                best_validation_result={
                    "bbox_list": [],
                    "bbox_path_list": v_bbox_imgs
                }
            )

            train_loss_list = []
            validation_loss_list = []

            filename = '{}.h5'.format(int(time.time()))

            for e in range(self.total_epoch):
                start_t0 = time.time()
                # stopイベントがセットされたら学習を中断する
                if self.stop_event.is_set():
                    return

                self.last_epoch = e

                epoch_id = storage.register_epoch(
                    model_id=self.model_id,
                    nth_epoch=e
                )

                train_loss = 0
                validation_loss = 0
                # Train
                batch_length = int(np.ceil(len(train_distributor)/float(self.batch_size)))

                i = 0
                self.last_batch = 0
                self.total_batch = batch_length
                for i, (train_x, train_y) in enumerate(train_distributor.batch(self.batch_size, True)):
                    start_t2 = time.time()

                    if self.stop_event.is_set():
                        return

                    self.model.set_models(inference=False)
                    h = self.model.freezed_forward(train_x/255. * 2 - 1)
                    with self.model.train():
                        z = self.model(h)
                        label = self.model.transform_label_format(train_y)
                        loss = self.model.loss_func(z, label)
                        num_loss = loss.as_ndarray().astype(np.float64)
                        loss += self.model.weight_decay()
                    # TODO: Specify batch length
                    loss.grad().update(self.model.optimizer(e, i,
                                       self.total_epoch, batch_length))

                    train_loss += num_loss

                    self.last_batch = i
                    self.last_train_loss = float(num_loss)
                    self.running_state = TRAIN

                    if DEBUG:
                        print('##### {}/{} {}'.format(i, batch_length, e))
                        print('  train loss', num_loss)
                        print('  learning rate', self.model.optimizer(e, i, self.total_epoch, batch_length)._lr)
                        print('  took time {}[s]'.format(time.time() - start_t2))
                train_loss = train_loss / (i + 1)
                train_loss_list.append(train_loss)

                start_t1 = time.time()
                if self.stop_event.is_set():
                    return

                if validation_distributor:
                    self.running_state = VALID
                    validation_loss, v_iou, v_mAP, v_bbox = \
                        self.run_validation(validation_distributor)
                    validation_loss_list.append(validation_loss)

                if self.best_validation_loss is None or validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss
                    bbox_list_len = min(len(v_bbox), len(v_bbox_imgs))
                    validation_results = {
                        "bbox_list": v_bbox[:bbox_list_len],
                        "bbox_path_list": v_bbox_imgs[:bbox_list_len]
                    }
                    storage.update_model_best_epoch(self.model_id, e, v_iou,
                                                    v_mAP, filename, validation_results)
                    # modelのweightを保存する
                    if not os.path.isdir(WEIGHT_DIR):
                        os.makedirs(WEIGHT_DIR)
                    self.model.save(os.path.join(WEIGHT_DIR, filename))

                if self.stop_event.is_set():
                    return

                storage.update_model_loss_list(
                    model_id=self.model_id,
                    train_loss_list=train_loss_list,
                    validation_loss_list=validation_loss_list,
                )
                cur_time = time.time()
                print("epoch %d done. %f. took time %f [s], %f [s]" % (e, train_loss,
                                                                       cur_time - start_t0,
                                                                       cur_time - start_t1,
                                                                       ))

                # このエポック時点でのiouとmapをDBに保存する
                bbox_list_len = min(len(v_bbox), len(v_bbox_imgs))
                storage.update_epoch(
                    epoch_id=epoch_id,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    epoch_iou=v_iou,
                    epoch_map=v_mAP)

            storage.update_model_state(self.model_id, STATE_FINISHED)
        except Exception as e:
            self.error_msg = e.args[0]

    def run_validation(self, distributor):
        try:
            validation_loss = 0
            v_ious = []
            v_mAPs = []
            v_bbox = []
            self.model.set_models(inference=True)
            for i, (validation_x, validation_y) in enumerate(distributor.batch(self.batch_size, False)):
                if self.stop_event.is_set():
                    return
                # Validation
                h = self.model.freezed_forward(validation_x/255. * 2 - 1)
                z = self.model(h)
                label = self.model.transform_label_format(validation_y)
                loss = self.model.loss_func(z, label)
                validation_loss += loss.as_ndarray()

                bbox = self.model.get_bbox(z.as_ndarray())
                iou, mAP = self.get_iou_and_mAP(validation_y, bbox)
                v_ious.extend(iou)
                v_mAPs.append(mAP)
                v_bbox.extend(bbox)
                if self.batch_size*(i + 1) > 1024:
                    break

            v_iou = np.mean(v_ious)
            v_mAP = np.mean(v_mAPs)
            v_iou = float(0 if np.isnan(v_iou) else v_iou)
            v_mAP = float(0 if np.isnan(v_mAP) else v_mAP)
            v_bbox = v_bbox

            validation_loss = validation_loss / (i + 1)
            return validation_loss, v_iou, v_mAP, v_bbox
        except Exception as e:
            self.error_msg = e.args[0]

    def set_train_config(self, class_num):
        try:
            self._class_num = int(class_num)
            return YoloDarknet(cell=self.cell_h, bbox=self.num_bbox, class_num=class_num, img_size=self.img_size)
        except Exception as e:
            self.error_msg = e.args[0]

    def stop(self):
        try:
            self.stop_event.set()
            storage.update_model_state(self.model_id, STATE_FINISHED)
        except Exception as e:
            self.error_msg = e.args[0]
