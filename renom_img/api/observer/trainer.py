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


# Generic Observer class
class TrainObserverBase(ABC):

    @abstractmethod
    def update_epoch(self,result): #called after training batch loop
        pass

    @abstractmethod
    def update_batch(self,result): # for training and validation batch update 
        pass

    @abstractmethod
    def start_train(self): # this is start point 
        pass

    @abstractmethod
    def start_valid(self): # called before starting validation batch loop
        pass

    @abstractmethod
    def start_evaluate(self): # called before starting evaluation
        pass

    @abstractmethod
    def end_train(self,train_result): # this is finish point
        pass

    @abstractmethod
    def end_valid(self,valid_result): # called after validation batch loop
        pass

    @abstractmethod
    def end_evaluate(self,evaluate_result): # called after finishing evaluation
        pass

class ObservableTrainer():

    def __init__(self, model):
        self.observers = []
        self.model = model

    def add_observer(self,observer):
        self.observers.append(observer)

    def remove_observer(self,observer):
        self.observers.remove(observer)

    def notify_update_epoch(self,result):
        for observer in self.observers:
            observer.update_epoch(result)

    def notify_update_batch(self,result):
        for observer in self.observers:
            observer.update_batch(result)

    def notify_start(self):
        for observer in self.observers:
            observer.start_train()

    def notify_start_valid(self):
        for observer in self.observers:
            observer.start_valid()

    def notify_start_evaluate(self):
        for observer in self.observers:
            observer.start_evaluate()

    def notify_end(self,train_result):
        for observer in self.observers:
            observer.end_train(train_result)

    def notify_end_valid(self,valid_result):
        for observer in self.observers:
            observer.end_valid(valid_result)

    def notify_end_evaluate(self,evaluate_result):
        for observer in self.observers:
            observer.end_evaluate(evaluate_result)


    def train(self,train_dist,valid_dist,total_epoch,batch_size,optimizer=None):

        avg_train_loss_list = []
        avg_valid_loss_list = []
        if optimizer is None:
            opt = self.model.default_optimizer
        else:
            opt = optimizer
        assert opt is not None
#         if opt is None:
#             raise InvalidParamError('Invalid optimizer {}.'.format(opt))

        train_batch_loop = len(train_dist)//batch_size

        if isinstance(opt, BaseOptimizer):
            opt.setup(train_batch_loop, total_epoch)
        # start of training
        self.notify_start()
        for epoch in range(total_epoch):
            if is_cuda_active():
                release_mem_pool()
            display_loss = 0

            for batch,val in enumerate(train_dist.batch(batch_size),1):
                if isinstance(self.model, Yolov2):
                    if is_cuda_active() and batch % 10 ==0:
                        release_mem_pool()
                    train_x, buffers, train_y=val[0],val[1],val[2]
                else:
                    train_x,train_y = val[0],val[1]

                self.model.set_models(inference=False)
                if len(train_x)>0:
                    with self.model.train():
                        if isinstance(self.model,Yolov2):
                            loss = self.model.loss(self.model(train_x),buffers,train_y)
                        else:
                            loss = self.model.loss(self.model(train_x),train_y)
                        reg_loss = loss + self.model.regularize()
                    reg_loss.grad().update(opt)
                    try:
                        loss = loss.as_ndarray()[0]
                    except:
                        loss = loss.as_ndarray()
                    loss = float(loss)
                    display_loss+=loss
                if isinstance(opt, BaseOptimizer):
                    opt.set_information(batch,epoch,avg_train_loss_list,avg_valid_loss_list,loss)
                # notify about batch update
                notify = {'batch':batch,'total_batch':train_batch_loop,'loss':loss}
                self.notify_update_batch(notify)

            avg_train_loss_list.append(display_loss/batch)
            # notify about epoch update
            notify = {'epoch':epoch,'total_epoch':total_epoch}
            self.notify_update_epoch(notify)

            #validation block
            valid_prediction,valid_target,avg_valid_loss = self.validation(valid_dist,batch_size)
            avg_valid_loss_list.append(avg_valid_loss)

            #Evaluation block
            self.evaluation(valid_prediction,valid_target)

        # end of training
        self.notify_end({'avg_train_loss_list':avg_train_loss_list,'avg_valid_loss_list':avg_valid_loss_list})

    def validation(self,valid_dist,batch_size):
        self.notify_start_valid()
        display_loss =0
        validation_prediction = []
        valid_target = []
        valid_batch_loop = int(np.ceil(len(valid_dist) / batch_size))
        for batch,val in enumerate(valid_dist.batch(batch_size,shuffle=False),1):
            if is_cuda_active():
                release_mem_pool()

            if isinstance(self.model, Yolov2):
                valid_x, buffers, valid_y=val[0],val[1],val[2]
            else:
                valid_x,valid_y = val[0],val[1]
            self.model.set_models(inference=True)

            valid_prediction_in_batch = self.model(valid_x)
            if isinstance(self.model,Yolov2):
                loss = self.model.loss(valid_prediction_in_batch,buffers,valid_y)
            else:
                loss = self.model.loss(valid_prediction_in_batch,valid_y)
            try:
                loss = loss.as_ndarray()[0]
            except:
                loss = loss.as_ndarray()
            loss = float(loss)
            display_loss += loss
            if isinstance(self.model,Classification):
                validation_prediction.append(rm.softmax(valid_prediction_in_batch).as_ndarray())
            else:
                validation_prediction.append(valid_prediction_in_batch.as_ndarray())
            if not isinstance(self.model,Detection):
                valid_target.append(valid_y)
            notify = {'batch':batch,'total_batch':valid_batch_loop,'loss':loss}
            self.notify_update_batch(notify)

        if isinstance(self.model,Detection):
            valid_target = valid_dist.get_resized_annotation_list(self.model.imsize)

        self.notify_end_valid({'avg_valid_loss':display_loss/batch})
        validation_prediction = np.concatenate(validation_prediction,axis=0)

        return validation_prediction,valid_target,(display_loss/batch)

    def evaluation(self,prediction,label):
        self.notify_start_evaluate()
        if isinstance(self.model,Classification):
            label = np.concatenate(label,axis=0)
            pred = np.argmax(prediction,axis=1)
            targ = np.argmax(label,axis=1)
            _, pr, _, rc, _, f1 = precision_recall_f1_score(pred,targ)
            prediction = [
                {
                    "score": [float(vc) for vc in v],
                    "class":float(p)
                }
                 for v, p in zip(prediction, pred)
            ]
            eval_matrix = {'precision':pr,'recall':rc,'f1':f1}
            self.notify_end_evaluate({'evaluation_matrix':eval_matrix,'prediction':prediction,'model':self.model})

        elif isinstance(self.model,Detection):

            n_valid = min(len(prediction),len(label))

            if isinstance(self.model,SSD):
                prediction_box = []
                for sample in range(n_valid):
                    prediction_b = self.model.get_bbox(np.expand_dims(prediction[sample],axis=0))
                    prediction_box.append(prediction_b[0])
            else:
                prediction_box = self.model.get_bbox(prediction[:n_valid])
            prec, rec, _, iou = get_prec_rec_iou(prediction_box,label[:n_valid])
            _, mAP = get_ap_and_map(prec, rec)

            eval_matrix = {'mAP':mAP,'iou':iou}
            self.notify_end_evaluate({'evaluation_matrix':eval_matrix,'prediction':prediction_box,'model':self.model})

        else: # for segmentation
            label = np.concatenate(label,axis=0)
            pred = np.argmax(prediction,axis=1)
            targ = np.argmax(label,axis=1)
            _, pr, _, rc, _, f1, _, _, _, _ = get_segmentation_metrics(pred, targ, n_class=model.num_class)
            prediction = []

            for p, t in zip(pred, targ):
                lep, lemp, ler, lemr, _, _, _, _, _, _ = get_segmentation_metrics(p[None],t[None],
                                                                                  n_class=model.num_class)
                prediction.append({
                    "class": p.astype(np.int).tolist(),
                    "recall": {k: float(v) for k, v in ler.items()},
                    "precision": {k: float(v) for k, v in lep.items()},
                })
            eval_matrix = {'precision':pr,'recall':rc,'f1':f1}
            self.notify_end_evaluate({'evaluation_matrix':eval_matrix,'prediction':prediction,'model':self.model})
