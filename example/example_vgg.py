import os
import matplotlib.pyplot as plt
import pandas as pd

from renom_img.api.classification.vgg import VGG16
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.augmentation import Augmentation
from renom_img.api.utility.augmentation.process import *
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.misc.display import draw_box

from renom.cuda import set_cuda_active
set_cuda_active(True)

# download dataset by api

image_caltech101 = "101_ObjectCategories"
class_map = sorted(os.listdir(image_caltech101))[1:]

image_path_list = []
label_list = []

for i, c in enumerate(class_map):
    root_path = os.path.join(image_caltech101, c)
    img_files = os.listdir(root_path)
    image_path_list.extend([os.path.join(root_path, path) for path in img_files])
    label_list += [i]*len(img_files)

N = len(image_path_list)
lin = np.arange(N)
train_N = int(N*0.5)

perm = np.random.permutation(train_N)
test_image_path_list = [image_path_list[p] for p in lin[train_N:]]
test_label_list = [label_list[p] for p  in lin[train_N:]]

train_image_path_list = [image_path_list[p] for p in lin[:train_N][perm[:int(train_N*0.8)]]]
train_label_list = [label_list[p] for p in lin[:train_N][perm[:int(train_N*0.8)]]]

valid_image_path_list = [image_path_list[p] for p in perm[:train_N][perm[int(train_N*0.8):]]]
valid_label_list = [label_list[p] for p in perm[:train_N][perm[int(train_N*0.8):]]]

model = VGG16(class_map, load_pretrained_weight=True, train_whole_network=False)

model.fit(train_image_path_list, train_label_list, valid_image_path_list, valid_label_list)

prediction_list = np.array([])
batch_size = 32
for i in range(0, len(test_image_path_list) // batch_size+1):
    prediction = model.predict(test_image_path_list[batch_size*i:batch_size*(i+1)])
    prediction_list = np.append(prediction_list, prediction)


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(prediction_list, test_label_list))
print(confusion_matrix(prediction_list, test_label_list))