# You need to specify the path to the model
# ex) python fine_tuning.py ./data/model/renom/***.h5

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
from renom_img.api.utility.augmentation import Augmentation
from renom_img.api.utility.augmentation.process import Flip, WhiteNoise, ContrastNorm
from PIL import Image

set_cuda_active(True)

def get_unique_label(lbl_list):
    uniq_label = []
    for i in tqdm.trange(len(lbl_list)):
        lbl_file = lbl_list[i]
        lbl = Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        Image.close()
        for l in set(lbl.flatten()):
            if not l in uniq_label:
                uniq_label.append(l)
    return uniq_label

def main():
    
    train_image_path_list = []
    train_annotation_path_list = []
    with open("./CamVid/train.txt") as f:
        txt = f.readlines()
        txt = [line.replace("/SegNet/","./").split(" ") for line in txt]
        for i in range(len(txt)):
            train_image_path_list.append(txt[i][0])
            train_annotation_path_list.append(txt[i][1].strip())
    valid_image_path_list = []
    valid_annotation_path_list = []
    with open("./CamVid/test.txt") as f:
        txt = f.readlines()
        txt = [line.replace("/SegNet/","./").split(" ") for line in txt]
        for i in range(len(txt)):
            valid_image_path_list.append(txt[i][0])
            valid_annotation_path_list.append(txt[i][1].strip())

    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    args = parser.parse_args()

    uniq_label = get_unique_label(train_annotation_path_list)
    uniq_label_test = get_unique_label(valid_annotation_path_list)
    for l in uniq_label_test:
        if not l in uniq_label:
            uniq_label.append(l)
    uniq_label.sort()
    from renom_img.api.segmentation.fcn import FCN8s
    model_class = FCN8s(np.arange(len(uniq_label)))
    lbl = model_class.get_label(valid_annotation_path_list, 0)
    check_class = 0
    lbl = (lbl==check_class) * 1.0
    visualize_img = draw_segment(valid_image_path_list[0], lbl.astype(np.int))
    visualize_img.save("./image.png")
    lbl = model_class.get_label(valid_annotation_path_list, 1)
    check_class = 0
    lbl = (lbl==check_class) * 1.0
    visualize_img = draw_segment(valid_image_path_list[1], lbl.astype(np.int))
    visualize_img.save("./image2.png")

    augmentation = Augmentation([
        Flip(),
        WhiteNoise(),
        ContrastNorm([0.5, 1.0])
    ])

    # Model
    model_class.load(args.model_file)
    model_class.set_models(inference=False)

    def callback_end_epoch(epoch, model, acg_train_loss_list, avg_valid_loss_list):
        if epoch % 10 == 0:
            model.save("./weight/model_{}.h5".format(epoch))

    model_class.fit(train_img_path_list=train_image_path_list,
                    train_annotation_list=train_annotation_path_list,
                    valid_img_path_list=valid_image_path_list,
                    valid_annotation_list=valid_annotation_path_list,
                    batch_size=8,
                    callback_end_epoch=callback_end_epoch,
                    augmentation=augmentation,
                    epoch=200)

    1/0
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
