import os
import shutil
import pytest
import time
import numpy as np
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET

from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.target import DataBuilderDetection
from renom_img.api.utility.augmentation import Augmentation
from renom_img.api.utility.augmentation.process import Shift, Rotate, RandomCrop, Jitter


def create_detection_test_files(root_dir, imsize):
    '''
    Create test images and xml data..
    '''
    os.makedirs(root_dir, exist_ok=True)
    for i in range(80):
        img = Image.fromarray(np.random.randint(0, 255, size=(*imsize, 3)).astype(np.uint8))
        img.save(os.path.join(root_dir, '{:03d}.png'.format(i)))
        xml_root = ET.Element('annotation')
        tree = ET.ElementTree(element=xml_root)
        xml_size = ET.SubElement(xml_root, 'size')
        xml_size_w = ET.SubElement(xml_size, 'width')
        xml_size_w.text = str(imsize[0])
        xml_size_h = ET.SubElement(xml_size, 'height')
        xml_size_h.text = str(imsize[1])
        for o in range(3):
            xml_obj = ET.SubElement(xml_root, 'object')
            xml_obj_name = ET.SubElement(xml_obj, 'name')
            xml_obj_name.text = "test"
            xml_obj_box = ET.SubElement(xml_obj, 'bndbox')
            xml_obj_box_xmin = ET.SubElement(xml_obj_box, 'xmin')
            xml_obj_box_xmin.text = '10'
            xml_obj_box_xmax = ET.SubElement(xml_obj_box, 'xmax')
            xml_obj_box_xmax.text = '50'
            xml_obj_box_ymin = ET.SubElement(xml_obj_box, 'ymin')
            xml_obj_box_ymin.text = '50'
            xml_obj_box_ymax = ET.SubElement(xml_obj_box, 'ymax')
            xml_obj_box_ymax.text = '150'
        tree.write("{}/{:03d}.xml".format(root_dir, i))


class Builder:

    def build_data(self, nth_epoch=10):
        self.nth_epoch = 10
        return self.builder

    def builder(self, img_path_list, annotation_list, augmentation, nth):
        img_list = []
        aa = self.nth_epoch
        for p in img_path_list:
            img = Image.open(p)
            img = np.asarray(img).transpose(2, 1, 0).copy()[None]
            img, _ = augmentation(img, annotation_list, mode="detection")
            img_list.append(img)
        return np.concatenate(img_list), annotation_list


imsize = (224 * 2, 224 * 2)


@pytest.mark.parametrize("builder", [
    # User defined builder.
    Builder().build_data(),
    # General builder.
    DataBuilderDetection(class_map=['test'], imsize=imsize).build,
    # Algorithm specific builder.
    Yolov1(class_map=['test'], imsize=imsize).build_data(),
])
def test_distributor_for_detection(builder):
    '''
    1. Create image.
    2. Create builder instance.
    3. Verify the distributor's output using builder.
    '''
    root_dir = 'text_dir'
    create_detection_test_files(root_dir, imsize)

    image_path_list = [os.path.join(root_dir, p) for p in os.listdir(
        root_dir) if os.path.splitext(p)[-1] == '.png']
    xml_path_list, class_map = parse_xml_detection(
        [os.path.join(root_dir, p) for p in os.listdir(root_dir) if os.path.splitext(p)[-1] == '.xml'])

    # Provides function to distributor.
    # Argumentation.
    aug = Augmentation([
        Shift(),
        Rotate(),
    ])
    dist = ImageDistributor(image_path_list, xml_path_list,
                            augmentation=aug, target_builder=builder, num_worker=4)
    start_t = time.time()
    for x, y in dist.batch(batch_size=8):
        assert x.shape == (8, 3, *imsize[::-1])
        time.sleep(0.01)
    print("Took time {:3.2f}[sec]".format(time.time() - start_t))

    # Remove files.
    shutil.rmtree(root_dir)
