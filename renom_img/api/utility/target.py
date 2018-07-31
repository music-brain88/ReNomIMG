#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from PIL import Image


"""Naming Rule.
- build_target_[algorithm name]

This type of function performs following transformation.
  annotation_list => algorithm specific target data

For confirming the format of annotation_list, 
see the function 'parse_xml_detection' written in load.py.
"""

RESIZE_METHOD = Image.BILINEAR


class DataBuilderBase(object):

    def __init__(self, class_map, imsize):
        if isinstance(imsize, int):
            imsize = (imsize, imsize)

        self.imsize = imsize
        self.class_map = class_map

    def __call__(self, img_path_list, annotation_list, augmentation=None, **kwargs):
        return self.build(img_path_list, annotation_list, augmentation, **kwargs)

    def build(self, img_path_list, annotation_list, augmentation=None, **kwargs):
        pass

    def reverce_label(self, label_list):
        pass

    def load_img(self, path):
        img = Image.open(path)
        img.load()
        w, h = img.size
        img = img.convert('RGB')
        img = img.resize(self.imsize, RESIZE_METHOD)
        img = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
        return img, self.imsize[0] / float(w), self.imsize[1] / h


class DataBuilderClassification(DataBuilderBase):
    """ Data builder for classification task

    Args:
        imsize(int or tuple): Input image size
        class_map(list): List of class id
   """

    def __init__(self, class_map, imsize):
        super(DataBuilderClassification, self).__init__(class_map, imsize)

    def build(self, img_path_list, annotation_list, augmentation=None, **kwargs):
        """
        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): List of class id
                                    [1, 4, 6 (int)]
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            x(ndarray): Batch of images
            y(ndarray): One hot labels for each image in a batch

        Example:
            >>> from renom_img.api.utility.target import DataBuilderClassification
            >>> builder = DataBuilderClassification(class_map, imsize)
            >>> x_train, y_train = builder.build(img_path_list, annotation_list)
       """

        # Check the class mapping.
        n_class = len(self.class_map)

        img_list = []
        label_list = []
        for img_path, an_data in zip(img_path_list, annotation_list):
            one_hot = np.zeros(n_class)
            img, sw, sh = self.load_img(img_path)
            img_list.append(img)
            one_hot[an_data] = 1.
            label_list.append(one_hot)
        if augmentation is not None:
            return augmentation(np.array(img_list), np.array(label_list), mode="classification")
        else:
            return np.array(img_list), np.array(label_list)


class DataBuilderDetection(DataBuilderBase):
    """ Data builder for detection task

    Args:
        imsize(int or tuple): Input image size
        class_map(list): List of class id
    """

    def build(self, img_path_list, annotation_list, augmentation=None, **kwargs):
        """
        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): List of annotations
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            (tuple):
                * x(ndarray): Batch of images
                * y(ndarray): The shape of ndarray is [# images, maximum number of objects in an image * (4(coordinates) + 1(confidence))]

        Example:
            >>> from renom_img.api.utility.target import DataBuilderDetection
            >>> builder = DataBuilderDetection(class_map, imsize)
            >>> x_train, y_train = builder.build(img_path_list, annotation_list)
        """
        # Check the class mapping.
        if self.class_map is None:
            class_dict = {}
            for annotation in annotation_list:
                for obj in annotation:
                    class_dict[obj['name']] = 1
            self.class_map = {k: i for i, k in enumerate(sorted(class_dict.keys()))}

        img_list = []
        new_annotation_list = []
        for i, img_path in enumerate(img_path_list):
            img, sw, sh = self.load_img(img_path)
            img_list.append(img)
            new_annotation_list.append([
                {
                    "box": [
                        an["box"][0] * sw,
                        an["box"][1] * sh,
                        an["box"][2] * sw,
                        an["box"][3] * sh,
                    ],
                    **{k: v for k, v in an.items() if k != 'box'}
                }
                for an in annotation_list[i]])

        if augmentation is not None:
            img_list, annotation_list = augmentation(
                np.array(img_list), new_annotation_list, mode="detection")
        else:
            img_list, annotation_list = np.array(img_list), new_annotation_list

        # Get max number of objects in one image.
        dlt = 4 + 1
        max_obj_num = np.max([len(annotation) for annotation in annotation_list])
        target = np.zeros((len(annotation_list), max_obj_num * dlt), dtype=np.float32)

        for i, annotation in enumerate(annotation_list):
            for j, obj in enumerate(annotation):
                target[i, j * dlt:(j + 1) * dlt - 1] = [obj['box'][0],
                                                        obj['box'][1], obj['box'][2], obj['box'][3]]
                target[i, (j + 1) * dlt - 1] = self.class_map[obj['name']]
        return img_list, target


class DataBuilderSegmentation(DataBuilderBase):
    """
    Annotation_list must be list of class name.

    """

    def build(self, img_path_list, annotation_path_list, augmentation=None, **kwargs):
        """
        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): List of annotation
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            (tuple):
                * x(ndarray): Batch of images
                * y(ndarray): The shape of ndarray is [# images, maximum number of objects in an image * (4(coordinates) + 1(confidence))]

        Example:
            >>> from renom_img.api.utility.target import DataBuilderSegmentation
            >>> builder = DataBuilderSegmentation(class_map, imsize)
            >>> x_train, y_train = builder.build(img_path_list, annotation_list)
        """
        # Check the class mapping.
        n_class = len(self.class_map)

        img_list = []
        label_list = []
        for img_path, an_path in zip(img_path_list, annotation_list):
            annot = np.zeros((n_class, self.imsize[0], self.imsize[1]))
            img, sw, sh = self.load_img(img_path)
            labels, asw, ash = self.load_img(an_path)[0]
            img_list.append(img)
            for i in range(self.imsize[0]):
                for j in range(self.imsize[1]):
                    annot[int(labels[i][j]), i, j] = 1
            label_list.append(annot)
        if augmentation is not None:
            return augmentation(np.array(img_list), np.array(label_list), mode="segmentation")
        else:
            return np.array(img_list), np.array(label_list)
