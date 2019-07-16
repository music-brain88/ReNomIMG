import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm

from renom_img.api import adddoc
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.load import load_img
from renom_img.api.utility.target import DataBuilderSegmentation
from renom_img.api.segmentation import SemanticSegmentation
from renom_img.api.cnn import CnnBase
from renom_img.api.utility.exceptions.exceptions import *


class CNN_UNet(CnnBase):

    def __init__(self, num_class=1):
        super(CNN_UNet, self).__init__()
        self.conv1_1 = rm.Conv2d(64, padding=1, filter=3)
        self.bn1_1 = rm.BatchNormalize(mode='feature')
        self.conv1_2 = rm.Conv2d(64, padding=1, filter=3)
        self.bn1_2 = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(128, padding=1, filter=3)
        self.bn2_1 = rm.BatchNormalize(mode='feature')
        self.conv2_2 = rm.Conv2d(128, padding=1, filter=3)
        self.bn2_2 = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(256, padding=1, filter=3)
        self.bn3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(256, padding=1, filter=3)
        self.bn3_2 = rm.BatchNormalize(mode='feature')
        self.conv4_1 = rm.Conv2d(512, padding=1, filter=3)
        self.bn4_1 = rm.BatchNormalize(mode='feature')
        self.conv4_2 = rm.Conv2d(512, padding=1, filter=3)
        self.bn4_2 = rm.BatchNormalize(mode='feature')
        self.conv5_1 = rm.Conv2d(1024, padding=1, filter=3)
        self.bn5_1 = rm.BatchNormalize(mode='feature')
        self.conv5_2 = rm.Conv2d(1024, padding=1, filter=3)
        self.bn5_2 = rm.BatchNormalize(mode='feature')

        self.deconv1 = rm.Deconv2d(512, stride=2)
        self.conv6_1 = rm.Conv2d(256, padding=1)
        self.conv6_2 = rm.Conv2d(256, padding=1)
        self.deconv2 = rm.Deconv2d(256, stride=2)
        self.conv7_1 = rm.Conv2d(128, padding=1)
        self.conv7_2 = rm.Conv2d(128, padding=1)
        self.deconv3 = rm.Deconv2d(128, stride=2)
        self.conv8_1 = rm.Conv2d(64, padding=1)
        self.conv8_2 = rm.Conv2d(64, padding=1)
        self.deconv4 = rm.Deconv2d(64, stride=2)
        self.conv9 = rm.Conv2d(num_class, filter=1)

    def forward(self, x):
        self._freeze()
        t = rm.relu(self.bn1_1(self.conv1_1(x)))
        c1 = rm.relu(self.bn1_2(self.conv1_2(t)))
        t = rm.max_pool2d(c1, filter=2, stride=2)
        t = rm.relu(self.bn2_1(self.conv2_1(t)))
        c2 = rm.relu(self.bn2_2(self.conv2_2(t)))
        t = rm.max_pool2d(c2, filter=2, stride=2)
        t = rm.relu(self.bn3_1(self.conv3_1(t)))
        c3 = rm.relu(self.bn3_2(self.conv3_2(t)))
        t = rm.max_pool2d(c3, filter=2, stride=2)
        t = rm.relu(self.bn4_1(self.conv4_1(t)))
        c4 = rm.relu(self.bn4_2(self.conv4_2(t)))
        t = rm.max_pool2d(c4, filter=2, stride=2)
        t = rm.relu(self.bn5_1(self.conv5_1(t)))
        t = rm.relu(self.bn5_2(self.conv5_2(t)))

        t = self.deconv1(t)[:, :, :c4.shape[2], :c4.shape[3]]
        t = rm.concat([c4, t])
        t = rm.relu(self.conv6_1(t))
        t = rm.relu(self.conv6_2(t))
        t = self.deconv2(t)[:, :, :c3.shape[2], :c3.shape[3]]
        t = rm.concat([c3, t])

        t = rm.relu(self.conv7_1(t))
        t = rm.relu(self.conv7_2(t))
        t = self.deconv3(t)[:, :, :c2.shape[2], :c2.shape[3]]
        t = rm.concat([c2, t])

        t = rm.relu(self.conv8_1(t))
        t = rm.relu(self.conv8_2(t))
        t = self.deconv4(t)[:, :, :c1.shape[2], :c1.shape[3]]
        t = rm.concat([c1, t])

        t = self.conv9(t)

        return t

    def _freeze(self):
        self.conv1_1.set_auto_update(self.train_whole)
        self.conv1_2.set_auto_update(self.train_whole)
        self.conv2_1.set_auto_update(self.train_whole)
        self.conv2_2.set_auto_update(self.train_whole)
        self.conv3_1.set_auto_update(self.train_whole)
        self.conv3_2.set_auto_update(self.train_whole)
        self.conv4_1.set_auto_update(self.train_whole)
        self.conv4_2.set_auto_update(self.train_whole)
        self.conv5_1.set_auto_update(self.train_whole)
        self.conv5_2.set_auto_update(self.train_whole)

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.conv9._channel = output_size
