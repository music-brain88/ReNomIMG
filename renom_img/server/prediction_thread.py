import os
import time
import numpy as np
import traceback
import csv
import xml.etree.ElementTree as et
from PIL import Image
from threading import Event
from renom.cuda import set_cuda_active, release_mem_pool

from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.detection.yolo_v2 import Yolov2
from renom_img.api.detection.ssd import SSD
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.augmentation.process import Shift

from renom_img.server.utility.storage import storage


class PredictionThread(object):
    pass
