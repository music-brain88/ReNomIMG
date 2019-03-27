import os
import sys
sys.setrecursionlimit(3000)
import renom as rm
import numpy as np
from tqdm import tqdm

from renom_img import __version__
from renom_img.api.utility.misc.download import download
from renom_img.api.classification import Classification
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification
from renom_img.api.classification import Classification
from renom_img.api.cnn import CnnBase


class InceptionV1Block(rm.Model):
    def __init__(self, channels=[64, 96, 128, 16, 32]):
        self.conv1 = rm.Conv2d(channels[0], filter=1)
        self.conv2_reduced = rm.Conv2d(channels[1], filter=1)
        self.conv2 = rm.Conv2d(channels[2], filter=3, padding=1)
        self.conv3_reduced = rm.Conv2d(channels[1], filter=1)
        self.conv3 = rm.Conv2d(channels[2], filter=5, padding=2)
        self.conv4 = rm.Conv2d(channels[1], filter=1)

    def forward(self, x):
        t1 = rm.relu(self.conv1(x))
        t2 = rm.relu(self.conv2_reduced(x))
        t2 = rm.relu(self.conv2(t2))
        t3 = rm.relu(self.conv3_reduced(x))
        t3 = rm.relu(self.conv3(t3))
        t4 = rm.max_pool2d(x, filter=3, stride=1, padding=1)
        t4 = rm.relu(self.conv4(t4))

        return rm.concat([t1, t2, t3, t4])


class CNN_InceptionV1(CnnBase):
    def __init__(self, num_class):
        super(CNN_InceptionV1, self).__init__()
        self.base1 = rm.Sequential([rm.Conv2d(64, filter=7, padding=3, stride=2),
                                    rm.Relu(),
                                    rm.MaxPool2d(filter=3, stride=2, padding=1),
                                    rm.BatchNormalize(mode='feature'),
                                    rm.Conv2d(64, filter=1, stride=1),
                                    rm.Relu(),
                                    rm.Conv2d(192, filter=3, padding=1, stride=1),
                                    rm.Relu(),
                                    rm.BatchNormalize(mode='feature'),
                                    rm.MaxPool2d(filter=3, stride=2, padding=1),
                                    InceptionV1Block(),
                                    InceptionV1Block([128, 128, 192, 32, 96, 64]),
                                    rm.MaxPool2d(filter=3, stride=2),
                                    InceptionV1Block([192, 96, 208, 16, 48, 64]),
                                    ])

        self.aux1 = rm.Sequential([rm.AveragePool2d(filter=5, stride=3),
                                   rm.Flatten(),
                                   rm.Dense(1024),
                                   rm.Dense(num_class)])

        self.base2 = rm.Sequential([InceptionV1Block([160, 112, 224, 24, 64, 64]),
                                    InceptionV1Block([128, 128, 256, 24, 64, 64]),
                                    InceptionV1Block([112, 144, 288, 32, 64, 64])])

        self.aux2 = rm.Sequential([rm.AveragePool2d(filter=5, stride=3),
                                   rm.Flatten(),
                                   rm.Dense(1024),
                                   rm.Dense(num_class)])

        self.base3 = rm.Sequential([InceptionV1Block([256, 160, 320, 32, 128, 128]),
                                    InceptionV1Block([256, 160, 320, 32, 128, 128]),
                                    InceptionV1Block([192, 384, 320, 48, 128, 128]),
                                    rm.AveragePool2d(filter=7, stride=1),
                                    rm.Flatten()])
        self.aux3 = rm.Dense(num_class)

    def forward(self, x):
        self._freeze()
        t = self.base1(x)
        out1 = self.aux1(t)
        t = self.base2(t)
        out2 = self.aux2(t)
        t = self.base3(t)
        out3 = self.aux3(t)
        return out1, out2, out3

    def _freeze(self):
        self.base1.set_auto_update(self.train_whole)
        self.base2.set_auto_update(self.train_whole)
        self.base3.set_auto_update(self.train_whole)

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.aux3._output_size = output_size
        self.aux1[-1]._output_size = output_size
        self.aux2[-1]._output_size = output_size


class InceptionV2BlockA(rm.Model):
    def __init__(self, channels=[64, 48, 64, 64, 96, 64]):
        self.conv1 = rm.Conv2d(channels[0], filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_reduced = rm.Conv2d(channels[1], filter=1)
        self.batch_norm2_reduced = rm.BatchNormalize(mode='feature')
        self.conv2 = rm.Conv2d(channels[2], filter=3, padding=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3_reduced = rm.Conv2d(channels[3], filter=1)
        self.batch_norm3_reduced = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(channels[4], filter=3, padding=1)
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(channels[4], filter=3, padding=1)
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(channels[5], filter=1)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1(self.conv1(x)))

        t2 = rm.relu(self.batch_norm2_reduced(self.conv2_reduced(x)))
        t2 = rm.relu(self.batch_norm2(self.conv2(t2)))
        t3 = rm.relu(self.batch_norm3_reduced(self.conv3_reduced(x)))
        t3 = rm.relu(self.batch_norm3_1(self.conv3_1(t3)))
        t3 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))

        t4 = rm.max_pool2d(x, filter=3, stride=1, padding=1)
        t4 = rm.relu(self.batch_norm4(self.conv4(t4)))

        return rm.concat([
            t1, t2, t3, t4
        ])


class InceptionV2BlockB(rm.Model):
    def __init__(self, channels=[64, 96, 384]):
        self.conv1_reduced = rm.Conv2d(channels[0], filter=1)
        self.batch_norm1_reduced = rm.BatchNormalize(mode='feature')
        self.conv1_1 = rm.Conv2d(channels[1], filter=3, padding=1)
        self.batch_norm1_1 = rm.BatchNormalize(mode='feature')
        self.conv1_2 = rm.Conv2d(channels[1], filter=3, stride=2)
        self.batch_norm1_2 = rm.BatchNormalize(mode='feature')

        self.conv2 = rm.Conv2d(channels[2], filter=3, stride=2)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1_reduced(self.conv1_reduced(x)))
        t1 = rm.relu(self.batch_norm1_1(self.conv1_1(t1)))
        t1 = rm.relu(self.batch_norm1_2(self.conv1_2(t1)))

        t2 = rm.relu(self.batch_norm2(self.conv2(x)))

        t3 = rm.max_pool2d(x, filter=3, stride=2)
        return rm.concat([t1, t2, t3])


class InceptionV2BlockC(rm.Model):
    def __init__(self, channels=[192, 128, 192, 128, 192, 192]):
        self.conv1 = rm.Conv2d(channels[0], filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_reduced = rm.Conv2d(channels[1], filter=1)
        self.batch_norm2_reduced = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(channels[1], filter=(3, 1), padding=(1, 0))
        self.batch_norm2_1 = rm.BatchNormalize(mode='feature')
        self.conv2_2 = rm.Conv2d(channels[2], filter=(1, 3), padding=(0, 1))
        self.batch_norm2_2 = rm.BatchNormalize(mode='feature')

        self.conv3_reduced = rm.Conv2d(channels[3], filter=1)
        self.batch_norm3_reduced = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(channels[3], filter=(3, 1), padding=(1, 0))
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(channels[3], filter=(1, 3), padding=(0, 1))
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')
        self.conv3_3 = rm.Conv2d(channels[3], filter=(3, 1), padding=(1, 0))
        self.batch_norm3_3 = rm.BatchNormalize(mode='feature')
        self.conv3_4 = rm.Conv2d(channels[4], filter=(1, 3), padding=(0, 1))
        self.batch_norm3_4 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(channels[5], filter=1)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1(self.conv1(x)))

        t2 = rm.relu(self.batch_norm2_reduced(self.conv2_reduced(x)))
        t2 = rm.relu(self.batch_norm2_1(self.conv2_1(t2)))
        t2 = rm.relu(self.batch_norm2_2(self.conv2_2(t2)))

        t3 = rm.relu(self.batch_norm3_reduced(self.conv3_reduced(x)))
        t3 = rm.relu(self.batch_norm3_1(self.conv3_1(t3)))
        t3 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))
        t3 = rm.relu(self.batch_norm3_3(self.conv3_3(t3)))
        t3 = rm.relu(self.batch_norm3_4(self.conv3_4(t3)))

        t4 = rm.max_pool2d(x, filter=3, stride=1, padding=1)
        t4 = rm.relu(self.batch_norm4(self.conv4(t4)))

        return rm.concat([
            t1, t2, t3, t4
        ])


class InceptionV2BlockD(rm.Model):
    def __init__(self, channels=[192, 320, 192, 192]):
        self.conv1_reduced = rm.Conv2d(channels[0], filter=1)
        self.batch_norm1_reduced = rm.BatchNormalize(mode='feature')
        self.conv1 = rm.Conv2d(channels[1], filter=3, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_reduced = rm.Conv2d(channels[2], filter=1)
        self.batch_norm2_reduced = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(channels[3], filter=3, padding=1)
        self.batch_norm2_1 = rm.BatchNormalize(mode='feature')
        self.conv2_2 = rm.Conv2d(channels[3], filter=3, stride=2)
        self.batch_norm2_2 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1_reduced(self.conv1_reduced(x)))
        t1 = rm.relu(self.batch_norm1(self.conv1(t1)))

        t2 = rm.relu(self.batch_norm2_reduced(self.conv2_reduced(x)))
        t2 = rm.relu(self.batch_norm2_1(self.conv2_1(t2)))
        t2 = rm.relu(self.batch_norm2_2(self.conv2_2(t2)))

        t3 = rm.max_pool2d(x, filter=3, stride=2)
        return rm.concat([
            t1, t2, t3
        ])


class InceptionV2BlockE(rm.Model):
    def __init__(self, channels=[320, 384, 384, 448, 384, 192]):
        self.conv1 = rm.Conv2d(channels[0], filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_reduced = rm.Conv2d(channels[1], filter=1)
        self.batch_norm2_reduced = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(channels[2], filter=(3, 1), padding=(1, 0))
        self.batch_norm2_1 = rm.BatchNormalize(mode='feature')
        self.conv2_2 = rm.Conv2d(channels[2], filter=(1, 3), padding=(0, 1))
        self.batch_norm2_2 = rm.BatchNormalize(mode='feature')

        self.conv3_reduced = rm.Conv2d(channels[3], filter=1)
        self.batch_norm3_reduced = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(channels[4], filter=3, padding=1)
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(channels[4], filter=(3, 1), padding=(1, 0))
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')
        self.conv3_3 = rm.Conv2d(channels[4], filter=(1, 3), padding=(0, 1))
        self.batch_norm3_3 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(channels[5], filter=1)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1(self.conv1(x)))

        t2 = rm.relu(self.batch_norm2_reduced(self.conv2_reduced(x)))
        t2_1 = rm.relu(self.batch_norm2_1(self.conv2_1(t2)))
        t2_2 = rm.relu(self.batch_norm2_2(self.conv2_2(t2)))
        t2 = rm.concat([t2_1, t2_2])

        t3 = rm.relu(self.batch_norm3_reduced(self.conv3_reduced(x)))
        t3 = rm.relu(self.batch_norm3_1(self.conv3_1(t3)))
        t3_1 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))
        t3_2 = rm.relu(self.batch_norm3_3(self.conv3_3(t3)))
        t3 = rm.concat([t3_1, t3_2])

        t4 = rm.max_pool2d(x, filter=3, stride=1, padding=1)
        t4 = rm.relu(self.batch_norm4(self.conv4(t4)))
        return rm.concat([
            t1, t2, t3, t4
        ])


class InceptionV2Stem(rm.Model):
    def __init__(self):
        self.conv1 = rm.Conv2d(32, filter=3, padding=0, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2 = rm.Conv2d(32, filter=3, padding=0, stride=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3 = rm.Conv2d(64, filter=3, padding=1, stride=1)
        self.batch_norm3 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(80, filter=3, stride=1)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')
        self.conv5 = rm.Conv2d(192, filter=3, stride=2)
        self.batch_norm5 = rm.BatchNormalize(mode='feature')
        self.conv6 = rm.Conv2d(192, filter=3, stride=1, padding=1)
        self.batch_norm6 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t = rm.relu(self.batch_norm1(self.conv1(x)))
        t = rm.relu(self.batch_norm2(self.conv2(t)))
        t = rm.relu(self.batch_norm3(self.conv3(t)))

        t = rm.max_pool2d(t, filter=3, stride=2)
        t = rm.relu(self.batch_norm4(self.conv4(t)))
        t = rm.relu(self.batch_norm5(self.conv5(t)))
        t = rm.relu(self.batch_norm6(self.conv6(t)))
        return t


class CNN_InceptionV3(CnnBase):
    def __init__(self, num_class):
        self.base1 = rm.Sequential([
            InceptionV2Stem(),
            InceptionV2BlockA([64, 48, 64, 64, 96, 32]),
            InceptionV2BlockA(),
            InceptionV2BlockA(),
            InceptionV2BlockB(),
            InceptionV2BlockC([192, 128, 192, 128, 192, 192]),
            InceptionV2BlockC(),
            InceptionV2BlockC(),
            InceptionV2BlockC()])

        self.aux1 = rm.Sequential([
            rm.AveragePool2d(filter=5, stride=3),
            rm.Conv2d(128, filter=1),
            rm.BatchNormalize(mode='feature'),
            rm.Relu(),
            rm.Conv2d(768, filter=1),
            rm.BatchNormalize(mode='feature'),
            rm.Relu(),
            rm.Flatten(),
            rm.Dense(num_class)])

        self.base2 = rm.Sequential([
            InceptionV2BlockD(),
            InceptionV2BlockE(),
            InceptionV2BlockE(),
            rm.AveragePool2d(filter=8),
            rm.Flatten()])

        self.aux2 = rm.Dense(num_class)

    def forward(self, x):
        self._freeze()
        t = self.base1(x)
        out1 = self.aux1(t)
        t = self.base2(t)
        out2 = self.aux2(t)

        return out1, out2

    def _freeze(self):
        self.base1.set_auto_update(self.train_whole)
        self.base2.set_auto_update(self.train_whole)

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.aux2._output_size = output_size
        self.aux1[-1]._output_size = output_size




class CNN_InceptionV2(CnnBase):
    def __init__(self, num_class):
        self.base1 = rm.Sequential([
            InceptionV2Stem(),
            InceptionV2BlockA([64, 48, 64, 64, 96, 32]),
            InceptionV2BlockA(),
            InceptionV2BlockA(),
            InceptionV2BlockB(),
            InceptionV2BlockC([192, 128, 192, 128, 192, 192]),
            InceptionV2BlockC(),
            InceptionV2BlockC(),
            InceptionV2BlockC()])
        self.aux1 = rm.Sequential([
            rm.AveragePool2d(filter=5, stride=3),
            rm.Conv2d(128, filter=1),
            rm.Relu(),
            rm.Conv2d(768, filter=1),
            rm.Relu(),
            rm.Flatten(),
            rm.Dense(num_class)])

        self.base2 = rm.Sequential([
            InceptionV2BlockD(),
            InceptionV2BlockE(),
            InceptionV2BlockE(),
            rm.AveragePool2d(filter=8),
            rm.Flatten()])

        self.aux2 = rm.Dense(num_class)

    def forward(self, x):
        self._freeze()
        t = self.base1(x)
        out1 = self.aux1(t)
        t = self.base2(t)
        out2 = self.aux2(t)

        return out1, out2

    def _freeze(self):
        self.base1.set_auto_update(self.train_whole)
        self.base2.set_auto_update(self.train_whole)

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.aux1[-1]._output_size = output_size
        self.aux2._output_size = output_size


class InceptionV4Stem(rm.Model):
    def __init__(self):
        self.conv1 = rm.Conv2d(32, filter=3, padding=0, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2 = rm.Conv2d(32, filter=3, padding=0, stride=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3 = rm.Conv2d(64, filter=3, padding=1, stride=1)
        self.batch_norm3 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(96, filter=3, stride=2)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')

        self.conv5_1_1 = rm.Conv2d(64, filter=1)
        self.batch_norm5_1_1 = rm.BatchNormalize(mode='feature')
        self.conv5_1_2 = rm.Conv2d(96, filter=3)
        self.batch_norm5_1_2 = rm.BatchNormalize(mode='feature')

        self.conv5_2_1 = rm.Conv2d(64, filter=1)
        self.batch_norm5_2_1 = rm.BatchNormalize(mode='feature')
        self.conv5_2_2 = rm.Conv2d(64, filter=(7, 1), padding=(3, 0))
        self.batch_norm5_2_2 = rm.BatchNormalize(mode='feature')
        self.conv5_2_3 = rm.Conv2d(64, filter=(1, 7), padding=(0, 3))
        self.batch_norm5_2_3 = rm.BatchNormalize(mode='feature')
        self.conv5_2_4 = rm.Conv2d(96, filter=3)
        self.batch_norm5_2_4 = rm.BatchNormalize(mode='feature')

        self.conv6 = rm.Conv2d(192, filter=3, stride=2)
        self.batch_norm6 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t = rm.relu(self.batch_norm1(self.conv1(x)))
        t = rm.relu(self.batch_norm2(self.conv2(t)))
        t = rm.relu(self.batch_norm3(self.conv3(t)))

        t1 = rm.max_pool2d(t, filter=3, stride=2)
        t2 = rm.relu(self.batch_norm4(self.conv4(t)))

        t = rm.concat([t1, t2])

        t1 = rm.relu(self.batch_norm5_1_1(self.conv5_1_1(t)))
        t1 = rm.relu(self.batch_norm5_1_2(self.conv5_1_2(t1)))

        t2 = rm.relu(self.batch_norm5_2_1(self.conv5_2_1(t)))
        t2 = rm.relu(self.batch_norm5_2_2(self.conv5_2_2(t2)))
        t2 = rm.relu(self.batch_norm5_2_3(self.conv5_2_3(t2)))
        t2 = rm.relu(self.batch_norm5_2_4(self.conv5_2_4(t2)))
        t = rm.concat([t1, t2])

        t1 = rm.relu(self.batch_norm6(self.conv6(t)))
        t2 = rm.max_pool2d(t, filter=3, stride=2)
        return rm.concat([t1, t2])


class InceptionV4BlockA(rm.Model):
    def __init__(self, channels=[64, 48, 64, 64, 96, 32]):
        self.conv1 = rm.Conv2d(96, filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_reduced = rm.Conv2d(64, filter=1)
        self.batch_norm2_reduced = rm.BatchNormalize(mode='feature')
        self.conv2 = rm.Conv2d(96, filter=3, padding=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3_reduced = rm.Conv2d(64, filter=1)
        self.batch_norm3_reduced = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(96, filter=3, padding=1)
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(96, filter=3, padding=1)
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(96, filter=1)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1(self.conv1(x)))

        t2 = rm.relu(self.batch_norm2_reduced(self.conv2_reduced(x)))
        t2 = rm.relu(self.batch_norm2(self.conv2(t2)))

        t3 = rm.relu(self.batch_norm3_reduced(self.conv3_reduced(x)))
        t3 = rm.relu(self.batch_norm3_1(self.conv3_1(t3)))
        t3 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))

        t4 = rm.max_pool2d(x, filter=3, stride=1, padding=1)
        t4 = rm.relu(self.batch_norm4(self.conv4(t4)))

        return rm.concat([
            t1, t2, t3, t4
        ])


class InceptionV4ReductionA(rm.Model):
    def __init__(self):
        # k, l, m, n
        # 192, 224, 256, 384
        self.conv1 = rm.Conv2d(384, filter=3, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_red = rm.Conv2d(192, filter=1)
        self.batch_norm2_red = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(224, filter=3, padding=1)
        self.batch_norm2_1 = rm.BatchNormalize(mode='feature')
        self.conv2_2 = rm.Conv2d(256, filter=3, stride=2)
        self.batch_norm2_2 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.max_pool2d(x, filter=3, stride=2)

        t2 = rm.relu(self.batch_norm1(self.conv1(x)))

        t3 = rm.relu(self.batch_norm2_red(self.conv2_red(x)))
        t3 = rm.relu(self.batch_norm2_1(self.conv2_1(t3)))
        t3 = rm.relu(self.batch_norm2_2(self.conv2_2(t3)))

        return rm.concat([
            t1, t2, t3
        ])


class InceptionV4BlockB(rm.Model):
    def __init__(self):
        self.conv1 = rm.Conv2d(128, filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2 = rm.Conv2d(384, filter=3, padding=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3_1 = rm.Conv2d(192, filter=1)
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(224, filter=(1, 7), padding=(0, 3))
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')
        self.conv3_3 = rm.Conv2d(256, filter=(7, 1), padding=(3, 0))
        self.batch_norm3_3 = rm.BatchNormalize(mode='feature')

        self.conv4_1 = rm.Conv2d(192, filter=1)
        self.batch_norm4_1 = rm.BatchNormalize(mode='feature')
        self.conv4_2 = rm.Conv2d(192, filter=(1, 7), padding=(0, 3))
        self.batch_norm4_2 = rm.BatchNormalize(mode='feature')
        self.conv4_3 = rm.Conv2d(224, filter=(7, 1), padding=(3, 0))
        self.batch_norm4_3 = rm.BatchNormalize(mode='feature')
        self.conv4_4 = rm.Conv2d(224, filter=(1, 7), padding=(0, 3))
        self.batch_norm4_4 = rm.BatchNormalize(mode='feature')
        self.conv4_5 = rm.Conv2d(256, filter=(7, 1), padding=(3, 0))
        self.batch_norm4_5 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.average_pool2d(x, filter=3, padding=1)
        t1 = rm.relu(self.batch_norm1(self.conv1(t1)))

        t2 = rm.relu(self.batch_norm2(self.conv2(x)))

        t3 = rm.relu(self.batch_norm3_1(self.conv3_1(x)))
        t3 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))
        t3 = rm.relu(self.batch_norm3_3(self.conv3_3(t3)))

        t4 = rm.relu(self.batch_norm4_1(self.conv4_1(x)))
        t4 = rm.relu(self.batch_norm4_2(self.conv4_2(t4)))
        t4 = rm.relu(self.batch_norm4_3(self.conv4_3(t4)))
        t4 = rm.relu(self.batch_norm4_4(self.conv4_4(t4)))
        t4 = rm.relu(self.batch_norm4_5(self.conv4_5(t4)))
        return rm.concat([t1, t2, t3, t4])


class InceptionV4ReductionB(rm.Model):
    def __init__(self):
        # k, l, m, n
        # 192, 224, 256, 384
        self.conv1_red = rm.Conv2d(192, filter=1)
        self.batch_norm1_red = rm.BatchNormalize(mode='feature')
        self.conv1 = rm.Conv2d(192, filter=3, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_red = rm.Conv2d(256, filter=1)
        self.batch_norm2_red = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(256, filter=(1, 7), padding=(0, 3))
        self.batch_norm2_1 = rm.BatchNormalize(mode='feature')
        self.conv2_2 = rm.Conv2d(320, filter=(7, 1), padding=(3, 0))
        self.batch_norm2_2 = rm.BatchNormalize(mode='feature')
        self.conv2_3 = rm.Conv2d(320, filter=3, stride=2)
        self.batch_norm2_3 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.max_pool2d(x, filter=3, stride=2)

        t2 = rm.relu(self.batch_norm1_red(self.conv1_red(x)))
        t2 = rm.relu(self.batch_norm1(self.conv1(t2)))

        t3 = rm.relu(self.batch_norm2_red(self.conv2_red(x)))
        t3 = rm.relu(self.batch_norm2_1(self.conv2_1(t3)))
        t3 = rm.relu(self.batch_norm2_2(self.conv2_2(t3)))
        t3 = rm.relu(self.batch_norm2_3(self.conv2_3(t3)))

        return rm.concat([
            t1, t2, t3
        ])


class InceptionV4BlockC(rm.Model):
    def __init__(self):
        self.conv1 = rm.Conv2d(256, filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2 = rm.Conv2d(256, filter=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3_red = rm.Conv2d(384, filter=1)
        self.batch_norm3_red = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(256, filter=(1, 3), padding=(0, 1))
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(256, filter=(3, 1), padding=(1, 0))
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')

        self.conv4_red = rm.Conv2d(384, filter=1)
        self.batch_norm4_red = rm.BatchNormalize(mode='feature')
        self.conv4_1 = rm.Conv2d(448, filter=(1, 3), padding=(0, 1))
        self.batch_norm4_1 = rm.BatchNormalize(mode='feature')
        self.conv4_2 = rm.Conv2d(512, filter=(3, 1), padding=(1, 0))
        self.batch_norm4_2 = rm.BatchNormalize(mode='feature')
        self.conv4_3 = rm.Conv2d(256, filter=(1, 3), padding=(0, 1))
        self.batch_norm4_3 = rm.BatchNormalize(mode='feature')
        self.conv4_4 = rm.Conv2d(256, filter=(3, 1), padding=(1, 0))
        self.batch_norm4_4 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.average_pool2d(x, filter=3, stride=1, padding=1)
        t1 = rm.relu(self.batch_norm1(self.conv1(t1)))

        t2 = rm.relu(self.batch_norm2(self.conv2(x)))

        t3 = rm.relu(self.batch_norm3_red(self.conv3_red(x)))
        t3_1 = rm.relu(self.batch_norm3_1(self.conv3_1(t3)))
        t3_2 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))

        t4 = rm.relu(self.batch_norm4_red(self.conv4_red(x)))
        t4 = rm.relu(self.batch_norm4_1(self.conv4_1(t4)))
        t4 = rm.relu(self.batch_norm4_2(self.conv4_2(t4)))
        t4_1 = rm.relu(self.batch_norm4_3(self.conv4_3(t4)))
        t4_2 = rm.relu(self.batch_norm4_4(self.conv4_4(t4)))

        return rm.concat([
            t1, t2, t3_1, t3_2, t4_1, t4_2
        ])


class CNN_InceptionV4(CnnBase):
    def __init__(self, num_class):

        self.block1 = rm.Sequential([InceptionV4Stem(),
                                     InceptionV4BlockA(),
                                     InceptionV4BlockA(),
                                     InceptionV4BlockA(),
                                     InceptionV4BlockA(),
                                     InceptionV4ReductionA()])
        self.block2 = rm.Sequential([
            InceptionV4BlockB(),
            InceptionV4BlockB(),
            InceptionV4BlockB(),
            InceptionV4BlockB(),
            InceptionV4BlockB(),
            InceptionV4BlockB(),
            InceptionV4BlockB(),
            InceptionV4ReductionB()])
        self.block3 = rm.Sequential([
            InceptionV4BlockC(),
            InceptionV4BlockC(),
            InceptionV4BlockC(),
            rm.AveragePool2d(filter=8),
            rm.Flatten(),
            rm.Dropout(0.2)
        ])

        self.fc = rm.Dense(num_class)

    def forward(self, x):
        self._freeze()
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.fc(t)
        return t

    def _freeze(self):
        self.block1.set_auto_update(self.train_whole)
        self.block2.set_auto_update(self.train_whole)
        self.block3.set_auto_update(self.train_whole)

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.fc._output_size = output_size
