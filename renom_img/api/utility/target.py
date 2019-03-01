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

    def resize_img(self, img_list, label_list):
        im_list = []

        for img in img_list:
            channel_last = img.transpose(1, 2, 0)
            img = Image.fromarray(np.uint8(channel_last))
            img = img.resize(self.imsize, RESIZE_METHOD).convert('RGB')
            im_list.append(np.asarray(img))

        return np.asarray(im_list).transpose(0, 3, 1, 2).astype(np.float32), np.asarray(label_list)

    def load_img(self, path):
        """ Loads an image

        Args:
            path(str): A path of an image

        Returns:
            (tuple): Returns image(numpy.array), the ratio of the given width to the actual image width,
                     and the ratio of the given height to the actual image height
        """
        img = Image.open(path)
        img.load()
        w, h = img.size
        img = img.convert('RGB')
        # img = img.resize(self.imsize, RESIZE_METHOD)
        img = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
        return img, self.imsize[0] / float(w), self.imsize[1] / h


class DataBuilderClassification(DataBuilderBase):
    """ Data builder for a classification task

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
    """

    def __init__(self, class_map, imsize):
        super(DataBuilderClassification, self).__init__(class_map, imsize)

    def build(self, img_path_list, annotation_list, augmentation=None, **kwargs):
        """ Builds an array of images and corresponding labels

        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): List of class id
                                    [1, 4, 6 (int)]
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            (tuple): Batch of images and corresponding one hot labels for each image in a batch
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
            img_list, label_list = augmentation(img_list, label_list, mode="classification")

        img_list, label_list = self.resize_img(img_list, label_list)

        return np.array(img_list), np.array(label_list)


class DataBuilderDetection(DataBuilderBase):
    """ Data builder for a detection task

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
    """

    def resize_img(self, img_list, annotation_list):
        im_list = []
        label_list = []

        for img, obj_list in zip(img_list, annotation_list):
            channel_last = img.transpose(1, 2, 0)
            img = Image.fromarray(np.uint8(channel_last))
            w, h = img.size
            sw, sh = self.imsize[0] / float(w), self.imsize[1] / float(h)
            img = img.resize(self.imsize, Image.BILINEAR).convert('RGB')
            new_obj_list = [{
                "box": [obj["box"][0] * sw, obj["box"][1] * sh, obj["box"][2] * sw, obj["box"][3] * sh],
                **{k: v for k, v in obj.items() if k != "box"}
            } for obj in obj_list]
            im_list.append(np.asarray(img))
            label_list.append(new_obj_list)
        return np.asarray(im_list).transpose(0, 3, 1, 2).astype(np.float32), label_list

    def build(self, img_path_list, annotation_list, augmentation=None, **kwargs):
        """
        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): The format of annotation list is as follows.
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            (tuple): Batch of images and ndarray whose shape is **(# images, maximum number of objects in an image * (4(coordinates) + 1(confidence)))**

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
                        an["box"][0],
                        an["box"][1],
                        an["box"][2],
                        an["box"][3],
                    ],
                    **{k: v for k, v in an.items() if k != 'box'}
                }
                for an in annotation_list[i]])

        if augmentation is not None:
            img_list, annotation_list = augmentation(
                img_list, new_annotation_list, mode="detection")

        img_list, annotation_list = self.resize_img(img_list, annotation_list)

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
    """ Data builder for a semantic segmentation task

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
    """

    def resize(self, img_list, label_list):
        x_list = []
        y_list = []
        for i, (img, label) in enumerate(zip(img_list, label_list)):
            channel_last = img.transpose(1, 2, 0)
            img = Image.fromarray(np.uint8(channel_last))
            img = img.resize(self.imsize, RESIZE_METHOD).convert('RGB')
            x_list.append(np.asarray(img))
            c, h, w = label.shape
            l = []
            for z in range(c):
                select = label[z, :, :]
                im = Image.fromarray(select)
                resized = im.resize(self.imsize, RESIZE_METHOD)
                l.append(np.array(resized))
            y_list.append(np.array(l))

        return np.asarray(x_list).transpose(0, 3, 1, 2).astype(np.float32), np.asarray(y_list)

    def load_annotation(self, path):
        """ Loads annotation data

        Args:
            path: A path of annotation file

        Returns:
            (tuple): Returns annotation data(numpy.array), the ratio of the given width to the actual image width,
        """
        N = len(self.class_map)
        img = Image.open(path)
        img.load()
        w, h = img.size
        # img = np.array(img.resize(self.imsize, RESIZE_METHOD))
        img = np.array(img)
        assert np.sum(np.histogram(img, bins=list(range(256)))[0][N:-1]) == 0
        assert img.ndim == 2
        return img, img.shape[0], img.shape[1]

    def load_img(self, path):
        img = Image.open(path)
        img.load()
        w, h = img.size
        img = img.convert('RGB')
        # img = img.resize(self.imsize, RESIZE_METHOD)
        img = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
        return img, w, h

    def crop_to_square(self, image):
        size = min(image.size)
        left, upper = (image.width - size) // 2, (image.height - size) // 2
        right, bottom = (image.width + size) // 2, (image.height + size) // 2
        return image.crop((left, upper, right, bottom))

    def build(self, img_path_list, annotation_list, augmentation=None, **kwargs):
        """
        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): The format of annotation list is as follows.
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            (tuple): Batch of images and ndarray whose shape is **(batch size, #classes, width, height)**

        """
        # Check the class mapping.
        n_class = len(self.class_map)

        img_list = []
        label_list = []
        for img_path, an_path in zip(img_path_list, annotation_list):
            img, sw, sh = self.load_img(img_path)
            labels, asw, ash = self.load_annotation(an_path)
            annot = np.zeros((n_class, asw, ash))
            img_list.append(img)
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if int(labels[i][j]) >= n_class:
                        annot[n_class - 1, i, j] = 1
                    else:
                        annot[int(labels[i][j]), i, j] = 1
            label_list.append(annot)
        if augmentation is not None:
            img_list, label_list = augmentation(img_list, label_list, mode="segmentation")
            return self.resize(img_list, label_list)
        else:
            return self.resize(img_list, label_list)
