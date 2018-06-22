import os
import sys
import numpy as np
import renom as rm



class UNet(rm.Model):
    """ U-Net: Convolutional Networks for Biomedical Image Segmentation

    Args:
        n_class (int): The number of classes

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> from renom_img.api.segmentation.unet import UNet
        >>> n, c, h, w = (2, 12, 64, 64)
        >>> x = rm.Variable(np.random.rand(n, c, h, w))
        >>> model = UNet(12)
        >>> t = model(x)
        >>> t.shape
        (2, 12, 64, 64)

    Note:
        Olaf Ronneberger, Philipp Fischer, Thomas Brox
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, n_class):
        self.conv1_1 = rm.Conv2d(64, padding=1, filter=3)
        self.conv1_2 = rm.Conv2d(64, padding=1, filter=3)
        self.conv2_1 = rm.Conv2d(128, padding=1, filter=3)
        self.conv2_2 = rm.Conv2d(128, padding=1, filter=3)
        self.conv3_1 = rm.Conv2d(256, padding=1, filter=3)
        self.conv3_2 = rm.Conv2d(256, padding=1, filter=3)
        self.conv4_1 = rm.Conv2d(512, padding=1, filter=3)
        self.conv4_2 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_1 = rm.Conv2d(1024, padding=1, filter=3)
        self.conv5_2 = rm.Conv2d(1024, padding=1, filter=3)

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
        self.conv9 = rm.Conv2d(n_class, filter=1)

    def forward(self, x):
        t = rm.relu(self.conv1_1(x))
        c1 = rm.relu(self.conv1_2(t))
        t = rm.max_pool2d(c1, filter=2, stride=2)
        t = rm.relu(self.conv2_1(t))
        c2 = rm.relu(self.conv2_2(t))
        t = rm.max_pool2d(c2, filter=2, stride=2)
        t = rm.relu(self.conv3_1(t))
        c3 = rm.relu(self.conv3_2(t))
        t = rm.max_pool2d(c3, filter=2, stride=2)
        t = rm.relu(self.conv4_1(t))
        c4 = rm.relu(self.conv4_2(t))
        t = rm.max_pool2d(c4, filter=2, stride=2)
        t = rm.relu(self.conv5_1(t))
        t = rm.relu(self.conv5_2(t))

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
