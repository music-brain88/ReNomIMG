from abc import ABC, abstractmethod
import renom as rm
from renom.cuda import release_mem_pool, is_cuda_active
import numpy as np
from renom_img.api.detection.yolo_v2 import Yolov2
from renom_img.api.detection.ssd import SSD
from renom_img.api.classification import Classification
from renom_img.api.detection import Detection
from renom_img.api.utility.optimizer import BaseOptimizer

from renom_img.api.utility.evaluate.segmentation import get_segmentation_metrics
from renom_img.api.utility.evaluate.classification import precision_recall_f1_score
from renom_img.api.utility.evaluate.detection import get_ap_and_map, get_prec_rec_iou
from renom_img.api.utility.exceptions.check_exceptions import *
from renom_img.api.utility.exceptions.exceptions import *


class TrainObserverBase(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def start_train_batches(self, notification):
        pass

    @abstractmethod
    def start_batch(self, result):
        pass

    @abstractmethod
    def end_batch(self, result):
        pass

    @abstractmethod
    def end_train_batches(self, notification):
        pass

    @abstractmethod
    def start_valid_batches(self):
        pass

    @abstractmethod
    def end_valid_batches(self, notification):
        pass

    @abstractmethod
    def start_evaluate(self):
        pass

    @abstractmethod
    def end_evaluate(self, evaluate_result):
        pass

    @abstractmethod
    def end(self, result):
        pass


class ObservableTrainer():

    def __init__(self, model):
        self.observers = []
        self.model = model
        self.stop_flag = False

    def add_observer(self, observer):
        if isinstance(observer, TrainObserverBase):
            self.observers.append(observer)
        else:
            raise InvalidTrainingThreadError(
                "Train thread must be an instance of TrainObserverBase class.")

    def remove_observer(self, observer):
        self.observers.remove(observer)

    def stop(self):
        self.stop_flag = True

    def notify_start(self):
        for observer in self.observers:
            observer.start()

    def notify_start_train_batches(self, notification):
        for observer in self.observers:
            observer.start_train_batches(notification)

    def notify_start_batch(self, notification):
        for observer in self.observers:
            observer.start_batch(notification)

    def notify_end_batch(self, notification):
        for observer in self.observers:
            observer.end_batch(notification)

    def notify_end_train_batches(self, notification):
        for observer in self.observers:
            observer.end_train_batches(notification)

    def notify_start_valid_batches(self):
        for observer in self.observers:
            observer.start_valid_batches()

    # notify start batch
    # notify end batch

    def notify_end_valid_batches(self, notification):
        for observer in self.observers:
            observer.end_valid_batches(notification)

    def notify_start_evaluate(self):
        for observer in self.observers:
            observer.start_evaluate()

    def notify_end_evaluate(self, evaluate_result):
        for observer in self.observers:
            observer.end_evaluate(evaluate_result)

    def notify_end(self, notification):
        for observer in self.observers:
            observer.end(notification)

    def train(self, train_dist, valid_dist, total_epoch, batch_size, optimizer=None):

        avg_train_loss_list = []
        avg_valid_loss_list = []
        if optimizer is None:
            opt = self.model.default_optimizer
        else:
            opt = optimizer
        if opt is None:
            raise InvalidOptimizerError(str("Provided Optimizer is not valid. Optimizer must be an instance of rm.optimizer. Provided {}".format(
                opt)))

        train_batch_loop = int(np.ceil(len(train_dist) / batch_size))

        if isinstance(opt, BaseOptimizer):
            opt.setup(train_batch_loop, total_epoch)

        # check stop training loop
        if self.stop_flag is True:
            return

        self.notify_start()
        for epoch in range(total_epoch):
            if is_cuda_active():
                release_mem_pool()
            display_loss = 0

            if self.stop_flag is True:
                return
            # start of training
            notify = {'epoch': epoch, 'total_epoch': total_epoch}
            self.notify_start_train_batches(notify)
            for batch, val in enumerate(train_dist.batch(batch_size), 1):
                if self.stop_flag is True:
                    return
                notify = {'batch': batch, 'total_batch': train_batch_loop}
                self.notify_start_batch(notify)
                if isinstance(self.model, Yolov2):
                    if is_cuda_active() and batch % 10 == 0:
                        release_mem_pool()
                    train_x, buffers, train_y = val[0], val[1], val[2]
                else:
                    train_x, train_y = val[0], val[1]

                self.model.set_models(inference=False)
                if (self.model._model.has_bn and len(train_x) > 1) or (not self.model._model.has_bn and len(train_x) > 0):
                    try:
                        with self.model.train():
                            if isinstance(self.model, Yolov2):
                                loss = self.model.loss(self.model(train_x), buffers, train_y)
                            else:
                                loss = self.model.loss(self.model(train_x), train_y)
                            reg_loss = loss + self.model.regularize()
                        reg_loss.grad().update(opt)
                    except Exception as ex:
                        if isinstance(ex, ReNomIMGError):
                            raise ex
                        else:
                            # in future we can check for memory usage here to raise OutofmemoryError explicitly
                            raise CudaError(str(ex))
                    if self.stop_flag is True:
                        return
                    try:
                        loss = loss.as_ndarray()[0]
                    except:
                        loss = loss.as_ndarray()
                    loss = float(loss)
                    # Exception checking
                    if np.isnan(loss):
                        raise InvalidLossValueError(
                            "Training has been stopped because the model parameters have become too large, causing a numerical overflow error.\n\n To prevent this, please try training again with a different algorithm or changing the following:\n Batch Size, \"Train Whole Network\" setting, \"Load pretrained weight\" setting or the number of images in your dataset.")

                    display_loss += loss
                    if isinstance(opt, BaseOptimizer):
                        opt.set_information(batch, epoch, avg_train_loss_list,
                                            avg_valid_loss_list, loss)
                    if self.stop_flag is True:
                        return
                    # notify about batch update
                    notify = {'loss': loss}
                    self.notify_end_batch(notify)

            avg_train_loss_list.append(display_loss / batch)
            # notify about batches end
            if self.stop_flag is True:
                return
            notify = {'avg_train_loss': display_loss / batch}
            self.notify_end_train_batches(notify)

            # validation block
            valid_prediction, valid_target, avg_valid_loss = self.validation(valid_dist, batch_size)
            if valid_prediction is None:
                return
            avg_valid_loss_list.append(avg_valid_loss)

            # Evaluation block
            self.evaluation(valid_prediction, valid_target)
            if self.stop_flag is True:
                return
        # end of training
        self.notify_end({'avg_train_loss_list': avg_train_loss_list,
                         'avg_valid_loss_list': avg_valid_loss_list})

    def validation(self, valid_dist, batch_size):
        self.notify_start_valid_batches()
        display_loss = 0
        validation_prediction = []
        valid_target = []
        valid_batch_loop = int(np.ceil(len(valid_dist) / batch_size))
        for batch, val in enumerate(valid_dist.batch(batch_size, shuffle=False), 1):
            if self.stop_flag is True:
                return None, None, None
            notify = {'batch': batch, 'total_batch': valid_batch_loop}
            self.notify_start_batch(notify)
            if is_cuda_active():
                release_mem_pool()

            if isinstance(self.model, Yolov2):
                valid_x, buffers, valid_y = val[0], val[1], val[2]
            else:
                valid_x, valid_y = val[0], val[1]
            self.model.set_models(inference=True)

            valid_prediction_in_batch = self.model(valid_x)
            try:
                if isinstance(self.model, Yolov2):
                    loss = self.model.loss(valid_prediction_in_batch, buffers, valid_y)
                else:
                    loss = self.model.loss(valid_prediction_in_batch, valid_y)
            except Exception as ex:
                if isinstance(ex, ReNomIMGError):
                    raise ex
                else:
                    raise CudaError(str(ex))
            try:
                loss = loss.as_ndarray()[0]
            except:
                loss = loss.as_ndarray()
            loss = float(loss)
            display_loss += loss
            if isinstance(self.model, Classification):
                validation_prediction.append(rm.softmax(valid_prediction_in_batch).as_ndarray())
            else:
                validation_prediction.append(valid_prediction_in_batch.as_ndarray())
            if not isinstance(self.model, Detection):
                valid_target.append(valid_y)
            if self.stop_flag is True:
                return None, None, None
            notify = {'loss': loss}
            self.notify_end_batch(notify)

        if isinstance(self.model, Detection):
            valid_target = valid_dist.get_resized_annotation_list(self.model.imsize)

        self.notify_end_valid_batches({'avg_valid_loss': display_loss / batch})
        if self.stop_flag is True:
            return
        validation_prediction = np.concatenate(validation_prediction, axis=0)

        return validation_prediction, valid_target, (display_loss / batch)

    def evaluation(self, prediction, label):
        self.notify_start_evaluate()
        if isinstance(self.model, Classification):
            label = np.concatenate(label, axis=0)
            pred = np.argmax(prediction, axis=1)
            targ = np.argmax(label, axis=1)
            _, pr, _, rc, _, f1 = precision_recall_f1_score(pred, targ)
            prediction = [
                {
                    "score": [float(vc) for vc in v],
                    "class":float(p)
                }
                for v, p in zip(prediction, pred)
            ]
            eval_matrix = {'precision': pr, 'recall': rc, 'f1': f1}
            self.notify_end_evaluate({'evaluation_matrix': eval_matrix,
                                      'prediction': prediction, 'model': self.model})

        elif isinstance(self.model, Detection):

            n_valid = min(len(prediction), len(label))

            if isinstance(self.model, SSD):
                prediction_box = []
                for sample in range(n_valid):
                    prediction_b = self.model.get_bbox(np.expand_dims(prediction[sample], axis=0))
                    prediction_box.append(prediction_b[0])
            else:
                prediction_box = self.model.get_bbox(prediction[:n_valid])
            prec, rec, _, iou = get_prec_rec_iou(prediction_box, label[:n_valid])
            _, mAP = get_ap_and_map(prec, rec)

            eval_matrix = {'mAP': mAP, 'iou': iou}
            self.notify_end_evaluate({'evaluation_matrix': eval_matrix,
                                      'prediction': prediction_box, 'model': self.model})

        else:  # for segmentation
            label = np.concatenate(label, axis=0)
            pred = np.argmax(prediction, axis=1)
            targ = np.argmax(label, axis=1)
            _, pr, _, rc, _, f1, _, _, _, _ = get_segmentation_metrics(
                pred, targ, n_class=self.model.num_class)
            prediction = []

            for p, t in zip(pred, targ):
                lep, lemp, ler, lemr, _, _, _, _, _, _ = get_segmentation_metrics(p[None], t[None],
                                                                                  n_class=self.model.num_class)
                prediction.append({
                    "class": p.astype(np.int).tolist(),
                    "recall": {k: float(v) for k, v in ler.items()},
                    "precision": {k: float(v) for k, v in lep.items()},
                })
            eval_matrix = {'precision': pr, 'recall': rc, 'f1': f1}
            self.notify_end_evaluate({'evaluation_matrix': eval_matrix,
                                      'prediction': prediction, 'model': self.model})
