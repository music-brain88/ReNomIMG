import renom as rm

from renom_img import __version__
from renom_img.api.cnn import CnnBase


class CnnYolov1(CnnBase):

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/Darknet.h5".format(__version__)

    def __init__(self, weight_decay=None):
        super(CnnYolov1, self).__init__()
        # Following is a darknet. We define network in each algorithm files.
        # This may cause duplicate definition but ensures independence.
        self.feature_extractor = rm.Sequential([
            # 1st Block
            rm.Conv2d(channel=64, filter=7, stride=2, padding=3, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 2nd Block
            rm.Conv2d(channel=192, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 3rd Block
            rm.Conv2d(channel=128, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 4th Block
            rm.Conv2d(channel=256, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 5th Block
            rm.Conv2d(channel=512, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
        ])
        self.classifier = rm.Sequential([
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, stride=2, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Flatten(),
            rm.Dense(4096),
            rm.LeakyRelu(slope=0.1),
            rm.Dropout(0.5),
            rm.Dense(1)
        ])

    def set_output_size(self, unit_size):
        # Need to set output_size here.
        self.output_size = unit_size
        self.classifier[-1]._output_size = unit_size

    def forward(self, x):
        '''
        '''
        # The attribute train_whole is defined in base class.
        self.feature_extractor.set_auto_update(self.train_whole)
        h = self.feature_extractor(x)
        return self.classifier(h)

    def reset_deeper_layer(self):
        for layer in self.classifier.iter_models():
            if isinstance(layer, rm.Parametrized):
                layer.params = {}

    def load_pretrained_weight(self, path):
        self.feature_extractor.load(path)
