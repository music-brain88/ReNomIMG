import renom as rm
import numpy as np


class StandardInit:
    def __init__(self):
        self.standards = {}
        self.build_standards()

    def build_standards(self):
        self.standards={'class_map':{
                            'value':[str],
                            'type':[type(None),list,dict],
                            'range':[1,10000]},
                        'imsize':{
                            'value':[int],
                            'type':[int,tuple],
                            'range':[16,2048]},
                        'load_pretrained_weight':{
                            'type':[bool,str]},
                        'train_whole_network':{
                            'type':[bool]},
                        'target_model':{
                            'type':rm.Model}

                       }

    def get_standards(self):
        return self.standards


class StandardResNetInit(StandardInit):
    def __init__(self):
        super(StandardResNetInit, self).__init__()
        self.build_standards()

    def build_standards(self):
        self.standards['plateau'] = {
            'type': [bool]}

class StandardResNeXtInit(StandardInit):
    def __init__(self):
        super(StandardResNextInit, self).__init__()
        self.build_standards()

    def build_standards(self):
        self.standards['plateau'] = {
            'type': [bool]}
        self.standards['cardinality'] = {
            'type': [int],
            'range': [1, 256]}


class StandardYolov1Init(StandardInit):
    def __init__(self):
        super(StandardYolov1Init, self).__init__()
        self.build_standards()

    def build_standards(self):
        self.standards['cells'] = {
            'type': [int, tuple],
            'range': [1, 20]}
        self.standards['bbox'] = {
            'type': [int],
            'range': [2, 20]}


class StandardYolov2Init(StandardInit):
    def __init__(self):
        super(StandardYolov2Init, self).__init__()
        self.build_standards()

    def build_standards(self):
        self.standards['imsize'] = {
                            'value':[int],
                            'type':[int,tuple],
                            'range':[32,2048]}
        self.standards['multiple'] = 32


class StandardSSDInit(StandardInit):
    def __init__(self):
        super(StandardSSDInit, self).__init__()
        self.build_standards()

    def build_standards(self):
        self.standards['overlap']={
                            'type':[float,np.float32,np.float64],
                            'range':[0.01,0.99]}
        self.standards['imsize']={
                            'type':300}

class StandardFCNInit(StandardInit):
    def __init__(self):
        super(StandardFCNInit, self).__init__()
        self.build_standards()

    def build_standards(self):
        self.standards['upscore'] = {
            'type': [bool]}

class StandardTernausNetInit(StandardInit):
    def __init__(self):
        super(StandardTernausNetInit, self).__init__()
        self.build_standards()

    def build_standards(self):
        self.standards['imsize'] = {
                            'value':[int],
                            'type':[int,tuple],
                            'range':[32,2048]}
        self.standards['multiple'] = 32

class StandardForward:
    def __init__(self):
        self.standards = {}
        self.build_standards()

    def build_standards(self):
        self.standards = {
            'value': [int, np.float32, np.float64],
            'type': [np.ndarray, rm.Variable],
            'length': 4}

    def get_standards(self):
        return self.standards


class StandardYolov2Forward(StandardForward):
    def __init__(self):
        super(StandardYolov2Forward, self).__init__()
        self.build_standards()

    def build_standards(self):
        self.standards['anchor']={
                            'value':[float,np.float32,np.float64],
                            'type':[np.ndarray],
                            'range':[2,20]} 

class StandardLR:
    def __init__(self):
        self.standards = {}
        self.build_standards()

    def build_standards(self):
        self.standards = {
                'LR':{
                    'type':[float,np.float32,np.float64],
                    'range':[0,1]}
                 }
    def get_standards(self):
        return self.standards

