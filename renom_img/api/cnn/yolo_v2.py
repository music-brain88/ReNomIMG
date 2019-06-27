import os
import sys
import numpy as np
import renom as rm
from renom_img import __version__
from renom_img.api.cnn import CnnBase
from renom_img.api.utility.exceptions.exceptions import *

class DarknetConv2dBN(rm.Model):

    def __init__(self, channel, filter=3, prev_ch=None):
        pad = int((filter - 1) / 2)
        if prev_ch is not None:
            self._conv = rm.Conv2d(channel=channel, filter=filter, padding=pad)
            self._conv.params = {
                "w": rm.Variable(self._conv._initializer((channel, prev_ch, filter, filter)), auto_update=True),
                "b": rm.Variable(np.zeros((1, channel, 1, 1), dtype=np.float32), auto_update=False),
            }
            self._bn = rm.BatchNormalize(mode='feature', momentum=0.99)
        else:
            self._conv = rm.Conv2d(channel=channel, filter=filter, padding=pad)
            self._bn = rm.BatchNormalize(mode='feature', momentum=0.99)

    def forward(self, x):

        return rm.leaky_relu(self._bn(self._conv(x)), 0.1)


class Darknet19Base(rm.Model):

    def __init__(self):
        self.block1 = rm.Sequential([
            DarknetConv2dBN(32, prev_ch=3),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block2 = rm.Sequential([
            DarknetConv2dBN(64, prev_ch=32),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block3 = rm.Sequential([
            DarknetConv2dBN(128, prev_ch=64),
            DarknetConv2dBN(64, filter=1, prev_ch=128),
            DarknetConv2dBN(128, prev_ch=64),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block4 = rm.Sequential([
            DarknetConv2dBN(256, prev_ch=128),
            DarknetConv2dBN(128, filter=1, prev_ch=256),
            DarknetConv2dBN(256, prev_ch=128),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block5 = rm.Sequential([
            DarknetConv2dBN(512, prev_ch=256),
            DarknetConv2dBN(256, filter=1, prev_ch=512),
            DarknetConv2dBN(512, prev_ch=256),
            DarknetConv2dBN(256, filter=1, prev_ch=512),
            DarknetConv2dBN(512, prev_ch=256),
        ])
        self.block6 = rm.Sequential([
            # For concatenation.
            rm.MaxPool2d(filter=2, stride=2),
            DarknetConv2dBN(1024, prev_ch=512),
            DarknetConv2dBN(512, filter=1, prev_ch=1024),
            DarknetConv2dBN(1024, prev_ch=512),
            DarknetConv2dBN(512, filter=1, prev_ch=1024),
            DarknetConv2dBN(1024, prev_ch=512),
        ])

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        f = self.block5(h)
        h = self.block6(f)
        return h, f


class CnnYolov2(CnnBase):

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/detection/Yolov2.h5".format(__version__)

    def __init__(self, weight_decay=None):
        super(CnnYolov2, self).__init__()
        self.class_map = None
        self.num_anchor= None
        self._base = Darknet19Base()
        self._conv1 = rm.Sequential([
            DarknetConv2dBN(channel=1024, prev_ch = 1024),
            DarknetConv2dBN(channel=1024, prev_ch = 1024),
        ])
        self._conv21 = DarknetConv2dBN(channel=64, prev_ch=512, filter=1)
        self._conv2 = DarknetConv2dBN(channel=1024, prev_ch= 1024 + 256)
        self._last = rm.Conv2d(channel=self.output_size, filter=1)
       
        for part in [self._conv21, self._conv1, self._conv2]:
            for layer in part.iter_models():
                if not layer.params:
                    continue
                if isinstance(layer, rm.Conv2d):
                    layer.params = {
                        "w": rm.Variable(layer._initializer(layer.params.w.shape), auto_update=True),
                        "b": rm.Variable(np.zeros_like(layer.params.b), auto_update=False),
                    }
                elif isinstance(layer, rm.BatchNormalize):
                    layer.params = {
                        "w": rm.Variable(layer._initializer(layer.params.w.shape), auto_update=True),
                        "b": rm.Variable(np.zeros_like(layer.params.b), auto_update=True),
                    }


    def set_output_size(self,out_size, class_map, num_anchor):
        self.class_map = class_map
        self.num_anchor = num_anchor
        self.output_size = out_size
        self._last._channel = out_size
        self._last.params = {
            "w": rm.Variable(self._last._initializer((self.output_size,1024,1,1)),auto_update=True),
            "b": rm.Variable(self._last._initializer((1,self.output_size,1,1)),auto_update=False),
        }
 

    def load_pretrained_weight(self,path):
        try:
            self._base.load(path)
        except:
            raise WeightLoadError('The pretrained weights path {} can not be loaded into the class {}.'.format(path,self.__class__))


    def reset_deeper_layer(self):
        pass

    def set_anchor(self, anchor_size):
        self.num_anchor = anchor_size 

    def forward(self, x):
        self._base.set_auto_update(self.train_whole)
        self._base.set_models(inference=(not self.train_whole or getattr(self, 'inference', False)))        
        h, f = self._base(x)
        f = self._conv21(f)
        h = self._conv1(h)

        h = self._conv2(rm.concat(h,rm.concat([f[:, :, i::2, j::2] for i in range(2) for j in range(2)])))
        out = self._last(h)

        # Create yolo format.
        N, C, H, W = h.shape

        reshaped = out.reshape(N, self.num_anchor, -1, W * H)
        conf = rm.sigmoid(reshaped[:, :, 0:1]).transpose(0, 2, 1, 3)
        px = rm.sigmoid(reshaped[:, :, 1:2]).transpose(0, 2, 1, 3)
        py = rm.sigmoid(reshaped[:, :, 2:3]).transpose(0, 2, 1, 3)
        pw = rm.exp(reshaped[:, :, 3:4]).transpose(0, 2, 1, 3)
        ph = rm.exp(reshaped[:, :, 4:5]).transpose(0, 2, 1, 3)
        cl = rm.softmax(reshaped[:, :, 5:].transpose(0, 2, 1, 3))
        return rm.concat(conf, px, py, pw, ph, cl).transpose(0, 2, 1, 3).reshape(N, -1, H, W)
        
