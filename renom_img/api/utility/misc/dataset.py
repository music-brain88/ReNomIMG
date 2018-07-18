import os
import numpy as np
import tarfile
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.load import parse_xml_detection


def fetch_detection_dataset_pets(split_validation=True, test_size=0.2):
    """

    Args:

    Returns:
        (list): This ret1urns list of image path.
        (list): This returns list of annotations.
            Each annotation has a list of dictionary which includes keys 'box' and 'name'.
            The structure is bellow.
        [
            [ # Objects of 1st image.
                {'box': [x(float), y, w, h], 'name': class_name(string), 'size': (x, y)(list of float), 'class': id(int)},
                {'box': [x(float), y, w, h], 'name': class_name(string), 'size': (x, y)(list of float), 'class': id(int)},
                ...
            ],
            [ # Objects of 2nd image.
                {'box': [x(float), y, w, h], 'name': class_name(string), 'size': image_size(float), 'class': id(int)},
               {'box': [x(float), y, w, h], 'name': class_name(string), 'size': image_size(float), 'class': id(int)},
                ...
            ]
        ]
    """
    pets_image_url = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    pets_label_url = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    pets_image_tar = "images.tar.gz"
    pets_label_tar = "annotations.tar.gz"

    setting = {
        "image": [pets_image_url, pets_image_tar],
        "label": [pets_label_url, pets_label_tar]
    }

    for url, path in setting.values():
        if not os.path.exists(path):
            download(url)

        with tarfile.open(path) as tar:
            tar.extractall(path="pets")

    def exist_xml_list(path, xml_name_list):
        name, _ = os.path.splitext(path)
        return name in xml_name_list

    xml_list = os.listdir("pets/annotations/xmls")
    xml_name_list = [os.path.splitext(name)[0] for name in xml_list]
    image_path_list = os.listdir("pets/images")
    image_name_list = [os.path.splitext(name)[0] for name in image_path_list]
    name_list = list(set(image_name_list) & set(xml_name_list))
    xml_list = [os.path.join("pets/annotations/xmls", name + ".xml") for name in name_list]
    annotation_list, _ = parse_xml_detection(xml_list)
    image_path_list = [os.path.join("pets/images", name + ".jpg") for name in name_list]

    if split_validation == False:
        return annotation_list, image_path_list

    else:
        image_path_list, annotation_list = np.array(image_path_list), np.array(annotation_list)
        indices = np.random.permutation(image_path_list.shape[0] - 1)
        threshold = int(np.round(test_size * image_path_list.shape[0]))
        train_index, test_index = indices[threshold:], indices[:threshold]
        train_annotation_list, valid_annotation_list = annotation_list[train_index], annotation_list[test_index]
        train_image_path_list, valid_image_path_list = image_path_list[train_index], image_path_list[test_index]
        return list(train_annotation_list), list(train_image_path_list), list(valid_annotation_list), list(valid_annotation_list)


def fetch_detection_dataset_voc_2007(split_validation=True):
    """

    Args:
        split_validation (boolean):
    Returns:
        (list): This ret1urns list of image path.
        (list): This returns list of annotations.
            Each annotation has a list of dictionary which includes keys 'box' and 'name'.
            The structure is bellow.
        [
            [ # Objects of 1st image.
                {'box': [x(float), y, w, h], 'name': class_name(string), 'size': (x, y)(list of float), 'class': id(int)},
                {'box': [x(float), y, w, h], 'name': class_name(string), 'size': (x, y)(list of float), 'class': id(int)},
                ...
            ],
            [ # Objects of 2nd image.
                {'box': [x(float), y, w, h], 'name': class_name(string), 'size': (x, y)(list of float), 'class': id(int)},
                {'box': [x(float), y, w, h], 'name': class_name(string), 'size': (x, y)(list of float), 'class': id(int)},
                ...
            ]
        ],
    """
    voc_2007_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"

    voc_2007_tar = "VOCtrainval_06-Nov-2007.tar"

    image_voc_2007 = "VOCdevkit/VOC2007/JPEGImages/"
    label_voc_2007 = "VOCdevkit/VOC2007/Annotations/"

    if not os.path.exists("VOCdevkit/VOC2007"):
        if not os.path.exists(voc_2007_tar):
            download(voc_2007_url)
        with tarfile.open(voc_2007_tar) as tar:
            tar.extractall()

    train_voc_2007 = [line.strip() for line in open(
        "VOCdevkit/VOC2007/ImageSets/Main/train.txt").readlines()]
    valid_voc_2007 = [line.strip() for line in open(
        "VOCdevkit/VOC2007/ImageSets/Main/val.txt").readlines()]

    train_image_path_list = []
    train_label_path_list = []
    valid_image_path_list = []
    valid_label_path_list = []

    # Use training dataset of VOC2007 as training data.
    for path in train_voc_2007:
        train_image_path_list.append(os.path.join(image_voc_2007, path + '.jpg'))
        train_label_path_list.append(os.path.join(label_voc_2007, path + '.xml'))

    # Use validation dataset of VOC2007 as validation data.
    for path in valid_voc_2007:
        valid_image_path_list.append(os.path.join(image_voc_2007, path + '.jpg'))
        valid_label_path_list.append(os.path.join(label_voc_2007, path + '.xml'))

    if split_validation == True:
        train_annotation_list, _ = parse_xml_detection(train_label_path_list)
        valid_annotation_list, _ = parse_xml_detection(valid_label_path_list)

        return train_annotation_list, train_image_path_list, valid_annotation_list, valid_image_path_list

    else:
        train_label_path_list.extend(valid_label_path_list)
        label_path_list = train_label_path_list
        train_image_path_list.extend(valid_image_path_list)
        image_path_list = train_image_path_list
        annotation_list, _ = parse_xml_detection(label_path_list)

        return annotation_list, image_path_list


def fetch_detection_dataset_voc_2012(split_validation=True):
    """

    Args:
        split_validation (boolean):
    Returns:
        (list): This ret1urns list of image path.
        (list): This returns list of annotations.
            Each annotation has a list of dictionary which includes keys 'box' and 'name'.
            The structure is bellow.
        [
            [ # Objects of 1st image.
                {'box': [x(float), y, w, h], 'name': class_name(string), 'size': (x, y)(list of float), 'class': id(int)},
                {'box': [x(float), y, w, h], 'name': class_name(string), 'size': (x, y)(list of float), 'class': id(int)},
                ...
            ],
            [ # Objects of 2nd image.
                {'box': [x(float), y, w, h], 'name': class_name(string), 'size': (x, y)(list of float), 'class': id(int)},
                {'box': [x(float), y, w, h], 'name': class_name(string), 'size': (x, y)(list of float), 'class': id(int)},
                ...
            ]
        ],
    """
    voc_2012_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

    voc_2012_tar = "VOCtrainval_11-May-2012.tar"

    image_voc_2012 = "VOCdevkit/VOC2012/JPEGImages/"
    label_voc_2012 = "VOCdevkit/VOC2012/Annotations/"

    if not os.path.exists("VOCdevkit/VOC2012"):
        if not os.path.exists(voc_2012_tar):
            download(voc_2012_url)
        with tarfile.open(voc_2012_tar) as tar:
            tar.extractall()

    train_voc_2012 = [line.strip() for line in open(
        "VOCdevkit/VOC2012/ImageSets/Main/train.txt").readlines()]
    valid_voc_2012 = [line.strip() for line in open(
        "VOCdevkit/VOC2012/ImageSets/Main/val.txt").readlines()]

    train_image_path_list = []
    train_label_path_list = []
    valid_image_path_list = []
    valid_label_path_list = []

    # Use training dataset of VOC2012 as training data.
    for path in train_voc_2012:
        train_image_path_list.append(os.path.join(image_voc_2012, path + '.jpg'))
        train_label_path_list.append(os.path.join(label_voc_2012, path + '.xml'))

    # Use validation dataset of VOC2012 as validation data.
    for path in valid_voc_2012:
        valid_image_path_list.append(os.path.join(image_voc_2012, path + '.jpg'))
        valid_label_path_list.append(os.path.join(label_voc_2012, path + '.xml'))

    if split_validation == True:
        train_annotation_list, _ = parse_xml_detection(train_label_path_list)
        valid_annotation_list, _ = parse_xml_detection(valid_label_path_list)

        return train_annotation_list, train_image_path_list, valid_annotation_list, valid_image_path_list

    else:
        train_label_path_list.extend(valid_label_path_list)
        label_path_list = train_label_path_list
        train_image_path_list.extend(valid_image_path_list)
        image_path_list = train_image_path_list
        annotation_list, _ = parse_xml_detection(label_path_list)

        return annotation_list, image_path_list

def fetch_classification_dataset_caltech101(split_validation=True, train_size=0.8):

    caltech101_url = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
    image_caltech101 = "101_ObjectCategories"
    caltech_101_tar = "101_ObjectCategories.tar.gz"

    if not os.path.exists(image_caltech101):
        download(caltech101_url)
        with tarfile.open(caltech_101_tar) as tar:
            tar.extractall()

    class_map = sorted(os.listdir(image_caltech101))

    image_path_list = []
    label_list = []

    for i, c in enumerate(class_map):
        root_path = os.path.join(image_caltech101, c)
        img_files = os.listdir(root_path)
        image_path_list.extend([os.path.join(root_path, path) for path in img_files])
        label_list += [i]*len(img_files)

    if split_validation == True:
        N = len(image_path_list)
        perm = np.random.permutation(N)
        train_N = int(N * train_size)

        train_image_path_list = [image_path_list[p] for p in perm[:train_N]]
        train_label_list = [label_list[p] for p in perm[:train_N]]

        valid_image_path_list = [image_path_list[p] for p in perm[train_N:]]
        valid_label_list = [label_list[p] for p in perm[train_N:]]

        return train_image_path_list, train_label_list, valid_image_path_list, valid_label_list
    else:
        return image_path_list, label_list

