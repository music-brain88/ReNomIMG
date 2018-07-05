import os
import time
import json
import numpy as np
from xml.etree import ElementTree
from multiprocessing import Pool
from tqdm import tqdm


def parse_xml(path, class_mapping_fname_id, class_mapping_name_id):
    tree = ElementTree.parse(path)
    root = tree.getroot()
    class_list = []
    for object_tree in root.findall('object'):
        class_name = object_tree.find('name').text
        try:
            class_list.append(class_mapping_fname_id[class_name])
        except Exception as e:
            class_list.append(class_mapping_name_id[class_name])
    return class_list


def load_file_train(image_net_path):
    class_mapping = json.load(open(os.path.join(image_net_path, "imagenet_class_index.json")))
    class_mapping_fname_id = {class_mapping[str(i)][0]:i for i in range(1000)}
    class_mapping_id_name = {i:class_mapping[str(i)][1] for i in range(1000)}
    class_mapping_name_id = {class_mapping[str(i)][1]:i for i in range(1000)}

    start_t = time.time()
    root1 = os.path.join(image_net_path, "img")
    img_path_list = [os.path.join(root1, path) for path in sorted(os.listdir(root1))] 
    root2 = os.path.join(image_net_path, "label")
    label_path_list = [os.path.join(root2, path) for path in sorted(os.listdir(root2))]

    print("Load img path list.", time.time() - start_t)

    N = len(img_path_list)
    y_array = np.zeros((N, 1000), np.int)
    start_t = time.time()
    seek = 0
    for n, img_path in enumerate(tqdm(img_path_list)):
        label_not_found_flag = True
        file_name = os.path.splitext(img_path.split("/")[-1])[0]
        class_id = class_mapping_fname_id[file_name.split("_")[0]]
        name = class_mapping_id_name[class_id]
        label_list = [class_id]
        label_path = label_path_list[min(seek, len(label_path_list)-1)]
        if file_name in label_path:
            label_list += parse_xml(label_path, class_mapping_fname_id, class_mapping_name_id)
            seek += 1
        label_list = list(set(label_list))
        assert len(label_list) < 2
        y_array[n, label_list[0]] = 1
    print("Parse time", time.time() - start_t)
    return img_path_list, y_array.astype(np.float32)
    

def load_file_valid(image_net_path):
    class_mapping = json.load(open(os.path.join(image_net_path, "imagenet_class_index.json")))
    class_mapping_fname_id = {class_mapping[str(i)][0]:i for i in range(1000)}
    class_mapping_id_name = {i:class_mapping[str(i)][1] for i in range(1000)}
    class_mapping_name_id = {class_mapping[str(i)][1]:i for i in range(1000)}

    start_t = time.time()
    root = os.path.join(image_net_path, "img")
    img_path_list = [os.path.join(root, path) for path in sorted(os.listdir(root))] 
    root = os.path.join(image_net_path, "label")
    label_path_list = [os.path.join(root, path) for path in sorted(os.listdir(root))]
    print("Load img path list.", time.time() - start_t)

    N = len(img_path_list)
    y_array = np.zeros((N, 1000), np.int)
    start_t = time.time()
    seek = 0
    for n, img_path in enumerate(tqdm(img_path_list)):
        label_not_found_flag = True
        file_name = os.path.splitext(img_path.split("/")[-1])[0]
        label_list = []
        label_path = label_path_list[seek]
        if file_name in label_path:
            label_list += parse_xml(label_path, class_mapping_fname_id, class_mapping_name_id)
            seek += 1
        label_list = list(set(label_list))
        assert len(label_list) == 1
        y_array[n, label_list[0]] = 1
    print("Parse time", time.time() - start_t)
    return img_path_list, y_array.astype(np.float32)

if __name__ == "__main__":
    load_file_valid("../dataset/val")
    load_file_train("../dataset/train")
