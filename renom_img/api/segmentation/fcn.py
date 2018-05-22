import os, sys
import numpy as np
import renom as rm

class FCN_Base(rm.Model):
    def __init__(self, nb_class, load_weight=False):
        self.conv1_1 = rm.Conv2d(64, padding=1, filter=3)
        self.conv1_2 = rm.Conv2d(64, padding=1, filter=3)
        self.max_pool1 = rm.MaxPool2d(filter=2, stride=2)

        self.conv2_1 = rm.Conv2d(128, padding=1, filter=3)
        self.conv2_2 = rm.Conv2d(128, padding=1, filter=3)
        self.max_pool2 = rm.MaxPool2d(filter=2, stride=2)

        self.conv3_1 = rm.Conv2d(256, padding=1, filter=3)
        self.conv3_2 = rm.Conv2d(256, padding=1, filter=3)
        self.conv3_3 = rm.Conv2d(256, padding=1, filter=3)
        self.max_pool3 = rm.MaxPool2d(filter=2, stride=2)

        self.conv4_1 = rm.Conv2d(512, padding=1, filter=3)
        self.conv4_2 = rm.Conv2d(512, padding=1, filter=3)
        self.conv4_3 = rm.Conv2d(512, padding=1, filter=3)
        self.max_pool4 = rm.MaxPool2d(filter=2, stride=2)

        self.conv5_1 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_2 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_3 = rm.Conv2d(512, padding=1, filter=3)
        self.max_pool5 = rm.MaxPool2d(filter=2, stride=2)

        self.fc6 = rm.Conv2d(4096, filter=7, padding=3)
        self.fc7 = rm.Conv2d(4096, filter=1)

class FCN_32s(FCN_Base):
    def __init__(self, nb_class, load_weight=False):
        super(FCN_32s, super).__init__(nb_class, load_weight)

        self.score_fr = rm.Conv2d(nb_class, filter=1) #nb_classes
        self.upscore = rm.Deconv2d(nb_class, stride=32, padding=0, filter=32) #nb_classes

    def forward(self, x):
        t = x
        t = rm.relu(self.conv1_1(t))
        t = rm.relu(self.conv1_2(t))
        t = self.max_pool1(t)

        t = rm.relu(self.conv2_1(t))
        t = rm.relu(self.conv2_2(t))
        t = self.max_pool2(t)

        t = rm.relu(self.conv3_1(t))
        t = rm.relu(self.conv3_2(t))
        t = rm.relu(self.conv3_3(t))
        t = self.max_pool3(t)

        t = rm.relu(self.conv4_1(t))
        t = rm.relu(self.conv4_2(t))
        t = rm.relu(self.conv4_3(t))
        t = self.max_pool4(t)

        t = rm.relu(self.conv5_1(t))
        t = rm.relu(self.conv5_2(t))
        t = rm.relu(self.conv5_3(t))
        t = self.max_pool5(t)

        t = rm.relu(self.fc6(t))
        fc6 = t

        t = rm.relu(self.fc7(t))
        fc7 = t

        t = self.score_fr(t)
        score_fr = t
        t = self.upscore(t)
        return t


class FCN_16s(FCN_Base):
    def __init__(self, nb_class, load_weight=False):
        super(FCN_16s, self).__init__(nb_class, load_weight)
        self.score_fr = rm.Conv2d(nb_class, filter=1)
        self.score_pool4 = rm.Conv2d(nb_class, filter=1)

        self.upscore2 = rm.Deconv2d(nb_class, filter=2, stride=2, padding=0)
        self.upscore16 = rm.Deconv2d(nb_class, filter=16, stride=16, padding=0)

    def forward(self, x):
        t = x
        t = rm.relu(self.conv1_1(t))
        t = rm.relu(self.conv1_2(t))
        t = self.max_pool1(t)

        t = rm.relu(self.conv2_1(t))
        t = rm.relu(self.conv2_2(t))
        t = self.max_pool2(t)

        t = rm.relu(self.conv3_1(t))
        t = rm.relu(self.conv3_2(t))
        t = rm.relu(self.conv3_3(t))
        t = self.max_pool3(t)

        t = rm.relu(self.conv4_1(t))
        t = rm.relu(self.conv4_2(t))
        t = rm.relu(self.conv4_3(t))
        t = self.max_pool4(t)
        pool4 = t

        t = rm.relu(self.conv5_1(t))
        t = rm.relu(self.conv5_2(t))
        t = rm.relu(self.conv5_3(t))
        t = self.max_pool5(t)
        t = rm.relu(self.fc6(t))

        t = rm.relu(self.fc7(t))

        t = self.score_fr(t)

        t = self.upscore2(t)
        upscore2 = t

        t = self.score_pool4(pool4)
        score_pool4 = t

        t = upscore2 + score_pool4
        fuse_pool4 = t
        t = self.upscore16(fuse_pool4)
        upscore16 = t

        return t

class FCN_8s(FCN_Base):
    def __init__(self, nb_class, load_weight=False):
        super(FCN_8s, self).__init__(nb_class, load_weight)
        self.drop_out = rm.Dropout(0.5)

        self.score_fr = rm.Conv2d(nb_class, filter=1)
        self.upscore2 = rm.Deconv2d(nb_class, filter=2, stride=2, padding=0)
        self.upscore8 = rm.Deconv2d(nb_class, filter=8, stride=8, padding=0)

        self.score_pool3 = rm.Conv2d(nb_class, filter=1)
        self.score_pool4 = rm.Conv2d(nb_class, filter=1)

        self.upscore_pool4 = rm.Deconv2d(nb_class, filter=2, stride=2, padding=0)

    def forward(self, x):
        t = x
        t = rm.relu(self.conv1_1(t))
        t = rm.relu(self.conv1_2(t))
        t = self.max_pool1(t) #112

        t = rm.relu(self.conv2_1(t))
        t = rm.relu(self.conv2_2(t))
        t = self.max_pool2(t) #56

        t = rm.relu(self.conv3_1(t))
        t = rm.relu(self.conv3_2(t))
        t = rm.relu(self.conv3_3(t))
        t = self.max_pool3(t) #28
        pool3 = t

        t = rm.relu(self.conv4_1(t))
        t = rm.relu(self.conv4_2(t))
        t = rm.relu(self.conv4_3(t))
        t = self.max_pool4(t)
        pool4 = t

        t = rm.relu(self.conv5_1(t))
        t = rm.relu(self.conv5_2(t))
        t = rm.relu(self.conv5_3(t))
        t = self.max_pool5(t)

        t = rm.relu(self.fc6(t))
        t = self.drop_out(t)
        fc6 = t

        t = rm.relu(self.fc7(t))
        fc7 = t

        t = self.score_fr(t)
        score_fr = t

        t = self.upscore2(t)
        upscore2 = t

        t = self.score_pool4(pool4)
        score_pool4 = t

        t = upscore2 + score_pool4
        fuse_pool4 = t

        t = self.score_pool3(pool3)
        score_pool3 = t


        t = self.upscore_pool4(fuse_pool4)
        upscore_pool4 = t
        t = upscore_pool4 + score_pool3

        t = self.upscore8(t)
        return t
