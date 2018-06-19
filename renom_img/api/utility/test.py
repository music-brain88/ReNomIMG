import numpy as np
import json
import pyximport; pyximport.install()
from evaluate import *
import pickle

class_list = ['backpack', 'bed', 'book', 'bookcase', 'bottle', 'bowl', 'cabinetry', 'chair', 'coffeetable', 'countertop', 'cup', 'diningtable', 'doll', 'door', 'heater', 'nightstand', 'person', 'pictureframe', 'pillow', 'pottedplant', 'remote', 'shelf', 'sink', 'sofa', 'tap', 'tincan', 'tvmonitor', 'vase', 'wastecontainer', 'windowblind', 'refrigerator', 'toilet', 'lamp', 'knife', 'oven','keyboard', 'toothbrush', 'laptop']
gt = pickle.load(open('./ground_truth.pkl'))
preds = pickle.load(open('./predictions.pkl'))

for g in gt:
    for ele in g:
        ele['class'] = class_list.index(ele['name'])
        del ele['name']
for p in preds:
    for ele in p:
        ele['class'] = class_list.index(ele['name'])
        del ele['name']

prec, recs = get_prec_and_rec(gt, preds)
#prec, recs, iou_per_cls, mean_iou = get_prec_rec_iou(gt, preds)
aps, mAP = get_ap_and_map(prec, recs)
#print(aps)
#print(mAP)
print(get_mean_iou(gt, preds))
