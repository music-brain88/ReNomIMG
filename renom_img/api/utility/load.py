import os
import numpy as np
from xml.etree import ElementTree
from concurrent.futures import ThreadPoolExecutor as Executor
from PIL import Image
from renom_img.api.utility.misc.display import draw_segment

def parse_xml_detection(xml_path_list, num_thread=8):
    """XML format must be Pascal VOC format.

    Args: 
        xml_path_list (list): List of xml-file's path.
        num_thread (int): Number of thread for parsing xml files.

    Returns:
        (list): This returns list of annotations.
        Each annotation has a list of dictionary which includes keys 'box' and 'name'.
        The structure is bellow.

    .. code-block :: python

        # An example of returned list.
        [
            [ # Objects of 1st image.
                {'box': [x(float), y, w, h], 'name': class_name(string), 'class': id(int)},
                {'box': [x(float), y, w, h], 'name': class_name(string), 'class': id(int)},
                ...
            ],
            [ # Objects of 2nd image.
                {'box': [x(float), y, w, h], 'name': class_name(string), 'class': id(int)},
                {'box': [x(float), y, w, h], 'name': class_name(string), 'class': id(int)},
                ...
            ]
        ]

    """

    global class_map
    annotation_list = []
    class_map = {}

    def load_thread(path_list):
        local_annotation_list = []
        for filename in path_list:
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
                image_data.append(
                    {'box': bounding_box, 'name': class_name, 'size': (width, height)})
            local_annotation_list.append(image_data)
        return local_annotation_list
    N = len(xml_path_list)
    if N > num_thread:
        batch = int(N / num_thread)
        with Executor(max_workers=num_thread + int(N % num_thread > 0)) as exc:
            ret = exc.map(load_thread, [xml_path_list[batch * i:batch * (i + 1)]
                                        for i in range(num_thread + int(N % num_thread > 0))])

        for r in ret:
            annotation_list += r
    else:
        annotation_list += load_thread(xml_path_list)

    class_map = [k for i, k in enumerate(sorted(class_map.keys()))]
    for annotation in annotation_list:
        for obj in annotation:
            obj["class"] = class_map.index(obj["name"])

    return annotation_list, class_map


def prepare_detection_data(img_path_list, annotation_list, imsize):
    N = len(img_path_list)
    img_list = []
    label_list = []

    for path, obj_list in zip(img_path_list, annotation_list):
        img = Image.open(path)
        w, h = img.size
        sw, sh = imsize[0] / float(w), imsize[1] / float(h)
        img = img.resize(imsize, Image.BILINEAR).convert('RGB')
        new_obj_list = [{
            "box": [obj["box"][0] * sw, obj["box"][1] * sh, obj["box"][2] * sw, obj["box"][3] * sh],
            **{k: v for k, v in obj.items() if k != "box"}
        } for obj in obj_list]
        img_list.append(np.asarray(img))
        label_list.append(new_obj_list)
    return np.asarray(img_list).transpose(0, 3, 1, 2).astype(np.float32), label_list


def parse_txt_classification(path, separator=" "):
    class_dict = {}
    filename_list = []
    annotation_list = []
    with open("path", "r") as reader:
        for line in reader.readlines():
            filename, classname = line.split(separator)
            class_dict[class_dict] = 1
            filename_list.append(filename)
            annotation_list.append(classname)

    class_map = [k for k, v in sorted(class_dict.items, key=lambda x: x[0])]
    annotation_list = [class_dict.index(a) for a in annotation_list]
    return annotation_list, class_map


def load_img(img_path, imsize=None):
    img = Image.open(img_path)
    img = img.convert('RGB')
    if imsize is not None:
        img = img.resize(imsize, Image.BILINEAR)
    return np.asarray(img).transpose(2, 0, 1).astype(np.float32)



def parse_classmap_file(class_map_file, separator=" "):
    """extract txt must be Pascal VOC format.

    Args: 
        file_path (poxis path): txt-file path.

    Returns:
        (list): This returns list of segmentation class list.
        Each segmentation has a list of dictionary which includes keys 'id' and 'value'.

    .. code-block :: python

        # An example of returned list.
        [
            [ # Objects of 1st line.
                {'id': index(int), 'value': class_name(string)}
            ],
            [ # Objects of 2nd line.
                {'id': index(int), 'value': class_name(string)}
            ]
        ]

    """
    class_map = list()
    with open(str(class_map_file)) as reader:
        for line in reader.readlines():
            class_name, id = line.split(separator)
            class_name.append({id: class_name})
    class_map = [c for k, c in sorted(class_map.items(), key=lambda x: x[0])]
    return class_map
