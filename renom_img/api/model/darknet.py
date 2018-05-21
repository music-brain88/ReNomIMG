import os
import sys
import numpy as np
import renom as rm


class Darknet(rm.Sequential):
    WEIGHT_URL = ""

    def __init__(self, last_unit_size, load_weight=False):
        super(Darknet, self).__init__([
            # 1st Block
            rm.Conv2d(channel=64, filter=7, stride=2, padding=3),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 2nd Block
            rm.Conv2d(channel=192, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 3rd Block
            rm.Conv2d(channel=128, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 4th Block
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 5th Block
            rm.Conv2d(channel=512, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, stride=2, padding=1),
            rm.LeakyRelu(slope=0.1),

            # 6th Block
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),

            # 7th Block
            rm.Flatten(),
            rm.Dense(1024),
            rm.LeakyRelu(slope=0.1),
            rm.Dense(4096),
            rm.LeakyRelu(slope=0.1),
            rm.Dropout(0.5),

            # 8th Block
            rm.Dense(last_unit_size),
        ])

        if load_weight:
            # Call download method.
            self.load('yolo.h5')
