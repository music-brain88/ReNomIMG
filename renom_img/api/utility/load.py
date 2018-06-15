import os
import numpy as np
from xml.etree import ElementTree
from PIL import Image


def parse_xml_detection(xml_path_list):
    """XML format must be Pascal VOC format.

    Args: 
        xml_path_list (list): List of xml-file's path.

    Returns:
        (list): This returns list of annotations.
            Each annotation has a list of dictionary which includes keys 'box' and 'name'.
            The structure is bellow.
        [
            [ # Objects of 1st image.
                {'box': [x(float), y, w, h], 'name': class_name(string)},
                {'box': [x(float), y, w, h], 'name': class_name(string)},
                ...
            ],
            [ # Objects of 2nd image.
                {'box': [x(float), y, w, h], 'name': class_name(string)},
                {'box': [x(float), y, w, h], 'name': class_name(string)},
                ...
            ]
        ]
    """
    annotation_list = []
    class_map = {}
    for filename in xml_path_list:
        tree = ElementTree.parse(filename)
        root = tree.getroot()
        size_tree = root.find('size')
        width = float(size_tree.find('width').text)
        height = float(size_tree.find('height').text)
        image_data = []
        for object_tree in root.findall('object'):
            bounding_box = object_tree.find('bndbox')
            xmin = float(bounding_box.find('xmin').text)
            ymin = float(bounding_box.find('ymin').text)
            xmax = float(bounding_box.find('xmax').text)
            ymax = float(bounding_box.find('ymax').text)
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + w / 2.
            y = ymin + h / 2.
            bounding_box = [x, y, w, h]
            class_name = object_tree.find('name').text.strip()
            class_map[class_name] = 1
            image_data.append({'box': bounding_box, 'name': class_name})
        annotation_list.append(image_data)
    class_map = {k:i for i, k in enumerate(sorted(class_map.keys()))}
    for annotation in annotation_list:
        for obj in annotation:
            obj["class"] = class_map[obj["name"]]
    return annotation_list, class_map

def prepare_detection_data(img_path_list, annotation_list, imsize):
    N = len(img_path_list)
    img_list = []
    label_list = []

    for path, obj_list in zip(img_path_list, annotation_list):
        img = Image.open(path)
        w, h = img.size
        sw, sh = imsize[0]/w, imsize[1]/h
        img = img.resize(imsize).convert('RGB')
        new_obj_list = [{
            "box": [obj["box"][0]*sw, obj["box"][1]*sh, obj["box"][2]*sw, obj["box"][3]*sh],
            "name": obj["name"],
            "class": obj["class"]
        } for obj in obj_list] 
        img_list.append(np.asarray(img))
        label_list.append(new_obj_list)
    return np.asarray(img_list).transpose(0, 3, 1, 2).astype(np.float32), label_list

def load_img(img_path, imsize=None):
    img = Image.open(img_path)
    if imsize is not None:
        img = img.resize(imsize, Image.BILINEAR)
    return np.asarray(img).transpose(2, 0, 1).astype(np.float32)
