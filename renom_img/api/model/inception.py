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


class InceptionV1(rm.Model):
    def __init__(self, n_class, load_weight=False):
        self.conv1 = rm.Conv2d(64, filter=7, padding=3, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')
        self.conv2 = rm.Conv2d(64, filter=1, stride=1)
        self.conv3 = rm.Conv2d(192, filter=3, padding=1, stride=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')
        self.a3 = InceptionV1Block()
        self.b3 = InceptionV1Block([128, 128, 192, 32, 96, 64])
        self.a4 = InceptionV1Block([192, 96, 208, 16, 48, 64])
        self.fc1_1 = rm.Dense(1024)
        self.fc1_2 = rm.Dense(n_class)
        self.b4 = InceptionV1Block([160, 112, 224, 24, 64, 64])
        self.c4 = InceptionV1Block([128, 128, 256, 24, 64, 64])
        self.d4 = InceptionV1Block([112, 144, 288, 32, 64, 64])
        self.fc2_1 = rm.Dense(1024)
        self.fc2_2 = rm.Dense(n_class)
        self.e4 = InceptionV1Block([256, 160, 320, 32, 128, 128])
        self.a5 = InceptionV1Block([256, 160, 320, 32, 128, 128])
        self.b5 = InceptionV1Block([192, 384, 320, 48, 128, 128])
        self.fc3 = rm.Dense(n_class)

        if load_weight:
            self.load('inceptionv1.h5')

    def forward(self, x):
        t = rm.relu(self.conv1(x))
        t = rm.max_pool2d(t, filter=3, stride=2, padding=1)
        t = self.batch_norm1(t)
        t = rm.relu(self.conv3(rm.relu(self.conv2(t))))
        t = self.batch_norm2(t)
        t = rm.max_pool2d(t, filter=3, stride=2, padding=1)
        t = self.a3(t)
        t = self.b3(t)
        t = rm.max_pool2d(t, filter=3, stride=2)
        t = self.a4(t)

        # 1st output -----------------
        out1 = rm.average_pool2d(t, filter=5, stride=3)
        out1 = rm.flatten(out1)
        out1 = self.fc1_1(out1)
        out1 = self.fc1_2(out1)
        # ----------------------------

        t = self.b4(t)
        t = self.c4(t)
        t = self.d4(t)

        # 2nd output ------------------
        out2 = rm.average_pool2d(t, filter=5, stride=3)
        out2 = rm.flatten(out2)
        out2 = self.fc2_1(out2)
        out2 = self.fc2_2(out2)
        # ----------------------------

        t = self.e4(t)
        t = self.a5(t)
        t = self.b5(t)
        t = rm.average_pool2d(t, filter=7, stride=1)
        t = rm.flatten(t)
        out3 = self.fc3(t)

        return out1, out2, out3

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

class InceptionV3(rm.Model):
    """
    Reference: https://arxiv.org/abs/1512.00567 -- Rethinking the Inception Architecture for Computer Vision
    """

    def __init__(self, n_classes=1000, load_weight=False):
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

        self.a1 = InceptionV2BlockA([64, 48, 64, 64, 96, 32])
        self.a2 = InceptionV2BlockA()
        self.a3 = InceptionV2BlockA()
        self.b1 = InceptionV2BlockB()

        self.c1 = InceptionV2BlockC([192, 128, 192, 128, 192, 192])
        self.c2 = InceptionV2BlockC()
        self.c3 = InceptionV2BlockC()
        self.c4 = InceptionV2BlockC()

        self.conv7 = rm.Conv2d(128, filter=1)
        self.batch_norm7 = rm.BatchNormalize(mode='feature')
        self.conv8 = rm.Conv2d(768, filter=5)
        self.batch_norm8 = rm.BatchNormalize(mode='feature')
        self.aux_fc = rm.Dense(n_classes)

        self.d1 = InceptionV2BlockD()

        self.e1 = InceptionV2BlockE()
        self.e2 = InceptionV2BlockE()
        self.fc = rm.Dense(n_classes)

        if load_weight:
            self.load('inceptionv3.h5')

    def forward(self, x):
        t = rm.relu(self.batch_norm1(self.conv1(x)))
        t = rm.relu(self.batch_norm2(self.conv2(t)))
        t = rm.relu(self.batch_norm3(self.conv3(t)))

        t = rm.max_pool2d(t, filter=3, stride=2)
        t = rm.relu(self.batch_norm4(self.conv4(t)))
        t = rm.relu(self.batch_norm5(self.conv5(t)))
        t = rm.relu(self.batch_norm6(self.conv6(t)))


        t = self.a1(t)
        t = self.a2(t)
        t = self.a3(t)

        t = self.b1(t)

        t = self.c1(t)
        t = self.c2(t)
        t = self.c3(t)
        t = self.c4(t)

        aux = rm.average_pool2d(t, filter=5, stride=3)
        aux = rm.relu(self.batch_norm7(self.conv7(aux)))
        aux = rm.relu(self.batch_norm8(self.conv8(aux)))
        aux = rm.flatten(aux)
        aux = self.aux_fc(aux)

        t = self.d1(t)
        t = self.e1(t)
        t = self.e2(t)
        t = rm.average_pool2d(t, filter=8)
        t = rm.flatten(t)
        t = self.fc(t)

        return t, aux


class InceptionV2(rm.Model):
    """
    Reference: https://arxiv.org/abs/1512.00567 -- Rethinking the Inception Architecture for Computer Vision
    """

    def __init__(self, n_classes=1000, load_weight=False):
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

        self.a1 = InceptionV2BlockA([64, 48, 64, 64, 96, 32])
        self.a2 = InceptionV2BlockA()
        self.a3 = InceptionV2BlockA()
        self.b1 = InceptionV2BlockB()

        self.c1 = InceptionV2BlockC([192, 128, 192, 128, 192, 192])
        self.c2 = InceptionV2BlockC()
        self.c3 = InceptionV2BlockC()
        self.c4 = InceptionV2BlockC()

        self.conv7 = rm.Conv2d(128, filter=1)
        self.conv8 = rm.Conv2d(768, filter=5)
        self.aux_fc = rm.Dense(n_classes)

        self.d1 = InceptionV2BlockD()

        self.e1 = InceptionV2BlockE()
        self.e2 = InceptionV2BlockE()
        self.fc = rm.Dense(n_classes)

        if load_weight:
            self.load('inceptionv2.h5')

    def forward(self, x):
        t = rm.relu(self.batch_norm1(self.conv1(x)))
        t = rm.relu(self.batch_norm2(self.conv2(t)))
        t = rm.relu(self.batch_norm3(self.conv3(t)))

        t = rm.max_pool2d(t, filter=3, stride=2)
        t = rm.relu(self.batch_norm4(self.conv4(t)))
        t = rm.relu(self.batch_norm5(self.conv5(t)))
        t = rm.relu(self.batch_norm6(self.conv6(t)))


        t = self.a1(t)
        t = self.a2(t)
        t = self.a3(t)

        t = self.b1(t)

        t = self.c1(t)
        t = self.c2(t)
        t = self.c3(t)
        t = self.c4(t)

        aux = rm.average_pool2d(t, filter=5, stride=3)
        aux = rm.relu(self.conv7(aux))
        aux = rm.relu(self.conv8(aux))
        aux = rm.flatten(aux)
        aux = self.aux_fc1(aux)

        t = self.d1(t)
        t = self.e1(t)
        t = self.e2(t)
        t = rm.average_pool2d(t, filter=8)
        t = rm.flatten(t)
        t = self.fc(t)

        return t, aux

