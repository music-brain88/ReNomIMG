import os
import numpy as np
from xml.etree import ElementTree


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
            image_data.append({'box': bounding_box, 'name': class_name})
        annotation_list.append(image_data)
    return annotation_list
