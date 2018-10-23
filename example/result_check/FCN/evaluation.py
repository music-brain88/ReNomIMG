# You need to specify the path to the model
# ex) python evaluation.py ./data/model/renom/***.h5

import argparse
import os
import os.path as osp
import re

import renom as rm
import numpy as np
import tqdm

from renom.cuda import set_cuda_active
from renom_img.api.utility.evaluate import Fast_Segmentation_Evaluator
from renom_img.api.utility.misc.display import draw_segment

set_cuda_active(True)

def main():
    class_map = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    
    valid_image_path_list = []
    valid_annotation_path_list = []
    with open("./external/fcn.berkeleyvision.org/data/pascal/seg11valid.txt") as f:
        txt = f.readlines()
        txt = [line.split("\n")[0] for line in txt]
        for i in range(len(txt)):
            valid_image_path_list.append("./VOCdevkit/VOC2012/JPEGImages/"+txt[i]+".jpg")
            valid_annotation_path_list.append("./VOCdevkit/VOC2012/SegmentationClass/"+txt[i]+".png")

    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    args = parser.parse_args()

    # Model
    basename = osp.basename(args.model_file).lower()
    match = re.match('^fcn(32|16|8)s.*$', basename)
    model_name = 'FCN%ss' % match.groups()[0]

    from renom_img.api.segmentation.fcn import FCN8s
    model_class = FCN8s(np.arange(len(class_map)))
    model_class.load(args.model_file)
    model_class.set_models(inference=True)

    # Forward
    if not osp.exists("./data/segmentation"):
        os.makedirs("./data/segmentation")
    lbl_preds, lbl_trues = [], []
    for i in tqdm.trange(len(valid_image_path_list)):
        datum = model_class.get_preprocessed_data(valid_image_path_list, i)
        x_data = np.expand_dims(datum, axis=0)
        lbl_true = model_class.get_label(valid_annotation_path_list, i)

        x = rm.Variable(x_data)
        score = model_class(x)
        lbl_pred = np.argmax(score.as_ndarray(), axis=1)

        lbl_preds.append(lbl_pred)
        lbl_trues.append(lbl_true)
        visualize_img = draw_segment(valid_image_path_list[i], lbl_pred[0])
        visualize_img.save("./data/segmentation/{}.png".format(str(i)))

    evaluator = Fast_Segmentation_Evaluator(lbl_preds, lbl_trues, class_map)
    acc, acc_cls, mean_iou, fwavacc = evaluator.evaluate()
    print("acc:{}, acc_cls:{}, mean_iou:{}, fwavacc:{}".format(\
        acc, acc_cls, mean_iou, fwavacc))
    print(evaluator.classification_report())

if __name__ == '__main__':
    main()
