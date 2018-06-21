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
        assert n_class == 1000 and load_weight, 'The number of classes must be 1000 if pre-trained weight is used'

        self.conv1 = rm.Conv2d(64, filter=7, padding=3, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')
        self.conv2 = rm.Conv2d(64, filter=1, stride=1)
        self.conv3 = rm.Conv2d(192, filter=3, padding=1, stride=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')
        self.a3 = InceptionBlock()
        self.b3 = InceptionBlock([128, 128, 192, 32, 96, 64])
        self.a4 = InceptionBlock([192, 96, 208, 16, 48, 64])
        self.fc1_1 = rm.Dense(1024)
        self.fc1_2 = rm.Dense(n_class)
        self.b4 = InceptionBlock([160, 112, 224, 24, 64, 64])
        self.c4 = InceptionBlock([128, 128, 256, 24, 64, 64])
        self.d4 = InceptionBlock([112, 144, 288, 32, 64, 64])
        self.fc2_1 = rm.Dense(1024)
        self.fc2_2 = rm.Dense(n_class)
        self.e4 = InceptionBlock([256, 160, 320, 32, 128, 128])
        self.a5 = InceptionBlock([256, 160, 320, 32, 128, 128])
        self.b5 = InceptionBlock([192, 384, 320, 48, 128, 128])
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



class Stem(rm.Model):
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


class InceptionV4(rm.Model):
    def __init__(self, n_class, load_weight=False):
        assert n_class == 1000 and load_weight, 'The number of classes must be 1000 if pre-trained weight is used'

        self.stem = Stem()

        self.a1 = InceptionV4BlockA()
        self.a2 = InceptionV4BlockA()
        self.a3 = InceptionV4BlockA()
        self.a4 = InceptionV4BlockA()

        self.a_red = InceptionV4ReductionA()

        self.b1 = InceptionV4BlockB()
        self.b2 = InceptionV4BlockB()
        self.b3 = InceptionV4BlockB()
        self.b4 = InceptionV4BlockB()
        self.b5 = InceptionV4BlockB()
        self.b6 = InceptionV4BlockB()
        self.b7 = InceptionV4BlockB()

        self.b_red = InceptionV4ReductionB()

        self.c1 = InceptionV4BlockC()
        self.c2 = InceptionV4BlockC()
        self.c3 = InceptionV4BlockC()

        self.dropout = rm.Dropout(0.2)
        self.fc = rm.Dense(n_class)

        if load_weight:
            self.load('inception_v4.h5')

    def forward(self, x):
        t = self.stem(x)
        t = self.a1(t)
        t = self.a2(t)
        t = self.a3(t)
        t = self.a4(t)

        t = self.a_red(t)

        t = self.b1(t)
        t = self.b2(t)
        t = self.b3(t)
        t = self.b4(t)
        t = self.b5(t)
        t = self.b6(t)
        t = self.b7(t)

        t = self.b_red(t)

        t = self.c1(t)
        t = self.c2(t)
        t = self.c3(t)

        t = rm.average_pool2d(t, filter=8)
        t = rm.flatten(t)
        t = self.dropout(t)

        t = self.fc(t)
        return t



