import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
from tqdm import tqdm
import renom as rm
from renom.cuda import set_cuda_active, release_mem_pool
from renom_img.api.utility.target import DataBuilderClassification
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.augmentation.augmentation import Augmentation
from renom_img.api.utility.augmentation.process import *

from darknet19 import Darknet19Classification
from image_net_utils import load_file_valid, load_file_train


set_cuda_active(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

aug = Augmentation([
        Shift(30, 30),
        Flip(),
        Rotate()
    ])

def load_for_classification(path):
    class_list = os.listdir(path)
    onehot_vectors = []
    for i in range(len(class_list)):
        temp = [0] * len(class_list)
        temp[i] = 1
        onehot_vectors.append(temp)
    X_list = []
    y_list = []
    for classname in class_list:
        imglist = os.listdir(path + classname)
        for filename in imglist:
            filepath = path + classname + "/" + filename
            X_list.append(filepath)
           # onehot = onehot_vectors[class_list.index(classname)]
            y_list.append(class_list.index(classname))

    return X_list, y_list, class_list

def get_class_from_onehot(onehot_vector, class_list):
    return class_list[onehot_vector.index(1)]

path = "/home/yamada/dataset/101_ObjectCategories/"
X_list, Y_list, class_list = load_for_classification(path)

N = len(X_list)
train_img_list = X_list[:int(N*0.8)]
train_annotation_list = Y_list[:int(N*0.8)]
valid_img_list = X_list[int(N*0.8):]
valid_annotation_list = Y_list[int(N*0.8):]

train_dist = ImageDistributor(train_img_list, train_annotation_list)
valid_dist = ImageDistributor(valid_img_list, valid_annotation_list)

# Image preprocess is done in darknet object.
model = Darknet19Classification()

def weight_decay(model):
    reg = 0
    for layer in model.iter_models():
        if hasattr(layer.params, "w") and isinstance(layer, rm.Conv2d):
            reg += rm.sum(layer.params.w * layer.params.w)
    return reg * 0.0005


EPOCH = 160
BATCH = 32
opt = rm.Sgd(0.1, 0.9)
lr_list = [0.1]*60 + [0.01]*40 + [0.001]*40
train_loss_list = []
valid_loss_list = []

save_dir = "result"
os.makedirs(save_dir, exist_ok=True)
log = open(os.path.join(save_dir, "log.txt"), "a")
log.write("{}:Train starts.\n".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S")))



for e in range(EPOCH):
    bar = tqdm(range(int(np.ceil(len(train_img_list)//BATCH))))
    display_loss = 0
    opt._lr = lr_list[e]
    for i, (x, y) in enumerate(train_dist.batch(BATCH, target_builder=DataBuilderClassification((224, 224), class_list))):
        model.set_models(inference=False)
        with model.train():
            z = model(x/255.*2 - 1)
            loss = rm.softmax_cross_entropy(z, y)
            reg_loss = loss + weight_decay(model)
        reg_loss.grad().update(opt)
        display_loss += float(loss.as_ndarray()[0])
        bar.set_description("epoch: {:03d}, loss: {:5.3f}".format(e,
                            float(loss.as_ndarray()[0])))
        bar.update(1)
    display_loss /= (i+1)
    train_loss_list.append(display_loss)

    bar.total = int(np.ceil(len(valid_dist)//BATCH))
    bar.n = 0
    display_loss = 0
    for i, (x, y) in enumerate(valid_dist.batch(BATCH, target_builder=DataBuilderClassification((224, 224), class_list))):
        model.set_models(inference=True)
        z = model(x/255.*2 - 1)
        loss = rm.softmax_cross_entropy(z, y)
        display_loss += float(loss.as_ndarray()[0])
        bar.set_description("epoch: {:03d}, valid loss: {:5.3f}".format(e,
                            float(loss.as_ndarray()[0])))
        bar.update(1)
    display_loss /= (i+1)
    valid_loss_list.append(display_loss)
    msg = "epoch: {:03d}, avg loss: {:5.3f}, avg valid loss{:5.3f}"
    bar.set_description(msg.format(e, train_loss_list[-1], valid_loss_list[-1]))
    bar.update(0)
    bar.close()

    ## Log
    log.write("{}:\n".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
    log.write("  Epoch: {}".format(e))
    log.write("  Train loss:{}\n".format(train_loss_list[-1]))
    log.write("  Valid loss:{}\n".format(valid_loss_list[-1]))
