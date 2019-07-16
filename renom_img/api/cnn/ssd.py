from __future__ import print_function, division
import os
import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api.cnn import CnnBase
from renom_img import __version__
from renom_img.api import Base, adddoc
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.exceptions.exceptions import *


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    layers.append(rm.MaxPool2d(filter=2, stride=2))
    return rm.Sequential(layers)


class CNN_VGG16(rm.Model):

    def __init__(self, num_class=1000):
        self.block1 = layer_factory(channel=64, conv_layer_num=2)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=3)
        self.block4 = layer_factory(channel=512, conv_layer_num=3)
        self.block5 = layer_factory(channel=512, conv_layer_num=3)
        self.fc1 = rm.Dense(4096)
        self.dropout1 = rm.Dropout(dropout_ratio=0.5)
        self.fc2 = rm.Dense(4096)
        self.dropout2 = rm.Dropout(dropout_ratio=0.5)
        self.fc3 = rm.Dense(num_class)

    def forward(self, x):
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)
        t = rm.flatten(t)
        t = rm.relu(self.fc1(t))
        t = self.dropout1(t)
        t = rm.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.fc3(t)
        return t


class CnnSSD(CnnBase):
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/detection/SSD.h5".format(__version__)

    def __init__(self, weight_decay=None):
        super(CnnSSD, self).__init__()
        self.has_bn = False

        self._feature_extractor = CNN_VGG16()
        self._freezed_network = rm.Sequential([self._feature_extractor.block1,
                                               self._feature_extractor.block2])

        block3 = self._feature_extractor.block3
        self.conv3_1 = block3._layers[0]
        self.conv3_2 = block3._layers[2]
        self.conv3_3 = block3._layers[4]
        self.pool3 = rm.MaxPool2d(filter=2, stride=2, padding=1)

        self.norm = rm.L2Norm(20)

        block4 = self._feature_extractor.block4
        self.conv4_1 = block4._layers[0]
        self.conv4_2 = block4._layers[2]
        self.conv4_3 = block4._layers[4]
        self.pool4 = rm.MaxPool2d(filter=2, stride=2)

        block5 = self._feature_extractor.block5
        self.conv5_1 = block5._layers[0]
        self.conv5_2 = block5._layers[2]
        self.conv5_3 = block5._layers[4]
        self.pool5 = rm.MaxPool2d(filter=3, stride=1, padding=1)

        # =================================================
        # THOSE ARE USED AFTER OUTPUS ARE NORMALIZED
        self.fc6 = rm.Conv2d(channel=1024, filter=3, padding=6, dilation=6)  # relu
        self.fc7 = rm.Conv2d(channel=1024, filter=1, padding=0)

        self.conv8_1 = rm.Conv2d(channel=256, filter=1)
        self.conv8_2 = rm.Conv2d(channel=512, stride=2, filter=3, padding=1)

        self.conv9_1 = rm.Conv2d(channel=128, filter=1)
        self.conv9_2 = rm.Conv2d(channel=256, stride=2, filter=3, padding=1)

        self.conv10_1 = rm.Conv2d(channel=128, filter=1, padding=0)
        self.conv10_2 = rm.Conv2d(channel=256, padding=0, stride=1, filter=3)

        self.conv11_1 = rm.Conv2d(channel=128, filter=1, padding=0)
        self.conv11_2 = rm.Conv2d(channel=256, padding=0, stride=1, filter=3)

        num_priors = 4
        self.conv4_3_mbox_loc = rm.Conv2d(num_priors * 4, padding=1, filter=3)
        self.conv4_3_mbox_conf = rm.Conv2d(padding=1, filter=3)
        # =================================================

        # =================================================
        num_priors = 6
        self.fc7_mbox_loc = rm.Conv2d(num_priors * 4, padding=1)
        self.fc7_mbox_conf = rm.Conv2d(padding=1, filter=3)
        # =================================================

        # =================================================
        self.conv8_2_mbox_loc = rm.Conv2d(num_priors * 4, padding=1, filter=3)
        self.conv8_2_mbox_conf = rm.Conv2d(padding=1, filter=3)
        # =================================================

        # =================================================
        self.conv9_2_mbox_loc = rm.Conv2d(num_priors * 4, padding=1)
        self.conv9_2_mbox_conf = rm.Conv2d(padding=1, filter=3)
        # =================================================

        # =================================================
        num_priors = 4
        self.conv10_2_mbox_loc = rm.Conv2d(num_priors * 4, padding=1)
        self.conv10_2_mbox_conf = rm.Conv2d(padding=1, filter=3)
        # =================================================

        # =================================================
        num_priors = 4
        self.conv11_2_mbox_loc = rm.Conv2d(num_priors * 4, padding=1)
        self.conv11_2_mbox_conf = rm.Conv2d(padding=1, filter=3)
        # =================================================

    def forward(self, x):
        self._freezed_network.set_auto_update(self.train_whole)
        x = self._freezed_network(x)
        n = x.shape[0]
        t = x
        # Vgg 3rd Block
        t = rm.relu(self.conv3_1(t))
        t = rm.relu(self.conv3_2(t))
        t = rm.relu(self.conv3_3(t))
        t = self.pool3(t)

        # Vgg 4th Block
        t = rm.relu(self.conv4_1(t))
        t = rm.relu(self.conv4_2(t))
        t = rm.relu(self.conv4_3(t))

        # Normalize and compute location, confidence and priorbox aspect ratio
        conv4_norm = self.norm(t)

        conv4_norm_loc = self.conv4_3_mbox_loc(conv4_norm)
        conv4_norm_loc_flat = rm.flatten(conv4_norm_loc.transpose(0, 2, 3, 1))
        conv4_norm_conf = self.conv4_3_mbox_conf(conv4_norm)
        conv4_norm_conf_flat = rm.flatten(conv4_norm_conf.transpose(0, 2, 3, 1))

        t = self.pool4(t)

        # Vgg 5th Block
        t = rm.relu(self.conv5_1(t))
        t = rm.relu(self.conv5_2(t))
        t = rm.relu(self.conv5_3(t))
        t = self.pool5(t)

        # Vgg 6, 7th Block
        t = rm.relu(self.fc6(t))
        t = rm.relu(self.fc7(t))
        # Confirmed here.

        # Normalize and compute location, confidence and priorbox aspect ratio
        fc7_mbox_loc = self.fc7_mbox_loc(t)
        fc7_mbox_loc_flat = rm.flatten(fc7_mbox_loc.transpose(0, 2, 3, 1))

        fc7_mbox_conf = self.fc7_mbox_conf(t)
        fc7_mbox_conf_flat = rm.flatten(fc7_mbox_conf.transpose(0, 2, 3, 1))

        t = rm.relu(self.conv8_1(t))
        t = rm.relu(self.conv8_2(t))
        # Normalize and compute location, confidence and priorbox aspect ratio
        conv8_mbox_loc = self.conv8_2_mbox_loc(t)
        conv8_mbox_loc_flat = rm.flatten(conv8_mbox_loc.transpose(0, 2, 3, 1))

        conv8_mbox_conf = self.conv8_2_mbox_conf(t)
        conv8_mbox_conf_flat = rm.flatten(conv8_mbox_conf.transpose(0, 2, 3, 1))

        t = rm.relu(self.conv9_1(t))
        t = rm.relu(self.conv9_2(t))
        # Normalize and compute location, confidence and priorbox aspect ratio
        conv9_mbox_loc = self.conv9_2_mbox_loc(t)
        conv9_mbox_loc_flat = rm.flatten(conv9_mbox_loc.transpose(0, 2, 3, 1))

        conv9_mbox_conf = self.conv9_2_mbox_conf(t)
        conv9_mbox_conf_flat = rm.flatten(conv9_mbox_conf.transpose(0, 2, 3, 1))

        t = rm.relu(self.conv10_1(t))
        t = rm.relu(self.conv10_2(t))

        conv10_mbox_loc = self.conv10_2_mbox_loc(t)
        conv10_mbox_loc_flat = rm.flatten(conv10_mbox_loc.transpose(0, 2, 3, 1))

        conv10_mbox_conf = self.conv10_2_mbox_conf(t)
        conv10_mbox_conf_flat = rm.flatten(conv10_mbox_conf.transpose(0, 2, 3, 1))

        t = rm.relu(self.conv10_1(t))
        t = rm.relu(self.conv10_2(t))

        conv11_mbox_loc = self.conv11_2_mbox_loc(t)
        conv11_mbox_loc_flat = rm.flatten(conv11_mbox_loc.transpose(0, 2, 3, 1))

        conv11_mbox_conf = self.conv11_2_mbox_conf(t)
        conv11_mbox_conf_flat = rm.flatten(conv11_mbox_conf.transpose(0, 2, 3, 1))

        mbox_loc = rm.concat([conv4_norm_loc_flat,
                              fc7_mbox_loc_flat,
                              conv8_mbox_loc_flat,
                              conv9_mbox_loc_flat,
                              conv10_mbox_loc_flat,
                              conv11_mbox_loc_flat])

        mbox_conf = rm.concat([conv4_norm_conf_flat,
                               fc7_mbox_conf_flat,
                               conv8_mbox_conf_flat,
                               conv9_mbox_conf_flat,
                               conv10_mbox_conf_flat,
                               conv11_mbox_conf_flat])

        mbox_loc = mbox_loc.reshape((n, -1, 4))
        mbox_conf = mbox_conf.reshape((n, -1, self.output_size))

        predictions = rm.concat([
            mbox_loc, mbox_conf
        ], axis=2)
        return predictions

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.conv4_3_mbox_conf._channel = 4 * output_size
        self.fc7_mbox_conf._channel = 6 * output_size
        self.conv8_2_mbox_conf._channel = 6 * output_size
        self.conv9_2_mbox_conf._channel = 6 * output_size
        self.conv10_2_mbox_conf._channel = 4 * output_size
        self.conv11_2_mbox_conf._channel = 4 * output_size

    def reset_deeper_layer(self):
        pass

    def load_pretrained_weight(self, path):
        try:
            self._feature_extractor.load(path)
        except:
            raise WeightLoadError(
                'The pretrained weights path {} can not be loaded into the class {}.'.format(path, self.__class__))
