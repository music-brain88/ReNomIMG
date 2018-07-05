import os
import numpy as np
import tqdm
import renom as rm
from renom.cuda import set_cuda_active
from darknet19 import Darknet19Detection, yolov2_loss
from utils import Yolov2Distributor, load_bbox, create_anchor

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
set_cuda_active(True)

img_list, annotation_list = load_bbox("../dataset/VOC2007/JPEGImages/", "../dataset/VOC2007/Annotations/")

base_size = (416, 416)
if not os.path.exists("anchor.txt"):
    box_list = []
    size_list = []
    for anot in annotation_list:
        for a in anot:
            box_list.append(a['bndbox'])
            size_list.append(a['size'])
    anchor = create_anchor(box_list, size_list, base_size=base_size)
    with open("anchor.txt", "w") as writer:
        for a in anchor:
            writer.write("{},{}\n".format(a[2], a[3]))
else:
    anchor = []
    with open("anchor.txt", "r") as reader:
        for r in reader.readlines():
            anchor.append([float(a) for a in r.strip().split(",")])
print(anchor)
dist = Yolov2Distributor(img_list, annotation_list)

model = Darknet19Detection()
opt = rm.Sgd(0.0001, 0.6)
for e in range(10):
    img_size = (224, 224)
    scale_w = img_size[0]/base_size[0]
    scale_h = img_size[1]/base_size[1]
    display_loss = 0
    bar = tqdm.tqdm()
    loss = 0
    for i, (x, y) in enumerate(dist.detection_batch(64, img_size=img_size)):
        with model.train():
            z = model(x/255.*2 - 1)
            target, mask = model.build_target(z.as_ndarray(), y,
                    [[a[0]*scale_w, a[1]*scale_h] for a in anchor], img_size)
            loss += rm.mse(z*mask, target)
        if i%2 != 0: continue
        loss.grad().update(opt)
        display_loss += loss.as_ndarray()
        bar.set_description("{}".format(loss.as_ndarray()))
        bar.update(1)
        loss = 0
    display_loss /= (i+1)
    print(display_loss)


