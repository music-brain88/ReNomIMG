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
        self.a3 = InceptionBlock()
        self.b3 = InceptionBlock([128, 128,192, 32, 96, 64])
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

        #2nd output ------------------
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
