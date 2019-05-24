import os
import shutil
import tarfile
from pathlib import Path
from xml.etree import ElementTree
from renom_img.api.utility.misc.download import download
from renom_img.server import create_directories, \
    DATASET_IMG_DIR, DATASET_LABEL_CLASSIFICATION_DIR, \
    DATASET_LABEL_DETECTION_DIR, DATASET_LABEL_SEGMENTATION_DIR, \
    DATASET_PREDICTION_DIR, DATASET_PREDICTION_IMG_DIR


def setup_example():
    """
    """
    print("### Setup a example dataset ###")
    create_directories()
    print("1/7: Downloading voc dataset.")
    voc2012 = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    if not os.path.exists("voc2012.tar"):
        download(voc2012, "voc2012.tar")

    print("2/7: Extracting tar file.")
    if not os.path.exists("voc"):
        with tarfile.open("voc2012.tar", "r:*") as tar:
            tar.extractall('voc')

    print("3/7: Moving image data to datasrc/img...")
    voc_img_dir = Path("voc/VOCdevkit/VOC2012/JPEGImages")
    for image in os.listdir(str(voc_img_dir)):
        shutil.copy(str(voc_img_dir / image), str(DATASET_IMG_DIR / image))

    print("4/7: Moving image data to datasrc/prediction_set/img...")
    for image in list(os.listdir(str(voc_img_dir)))[::50]:
        shutil.copy(str(voc_img_dir / image), str(DATASET_PREDICTION_IMG_DIR / image))

    # Setting Detection
    print("5/7: Moving xml data to datasrc/label/detection...")
    voc_detection_label_dir = Path("voc/VOCdevkit/VOC2012/Annotations")
    for xml in os.listdir(str(voc_detection_label_dir)):
        shutil.copy(str(voc_detection_label_dir / xml), str(DATASET_LABEL_DETECTION_DIR / xml))

    # Setting Segmentation
    print("6/7: Moving segmentation target data to datasrc/label/segmentation...")
    classes = list(sorted([
        "aeroplane", "bicycle", "boat", "bus", "car",
        "motorbike", "train", "person", "bird", "cat",
        "cow", "dog", "horse", "sheep", "bottle",
        "chair", "dining_table", "potted_plant", "sofa", "tv/monitor"
    ]))
    voc_segmentation_label_dir = Path("voc/VOCdevkit/VOC2012/SegmentationClass")
    for xml in os.listdir(str(voc_segmentation_label_dir)):
        shutil.copy(str(voc_segmentation_label_dir / xml),
                    str(DATASET_LABEL_SEGMENTATION_DIR / xml))
    with open(str(DATASET_LABEL_SEGMENTATION_DIR / "class_map.txt"), "w") as writer:
        for i, c in enumerate(["background"] + classes):
            writer.write("{} {}\n".format(c, i))

    # Setting Classification
    print("7/7: Creating classification target data to datasrc/label/classification...")
    with open(str(DATASET_LABEL_CLASSIFICATION_DIR / "target.txt"), "w") as writer:
        for path in sorted(voc_detection_label_dir.iterdir()):
            tree = ElementTree.parse(str(path))
            root = tree.getroot()
            obj = list(root.findall('object'))[0]
            class_name = obj.find('name').text.strip()
            writer.write("{} {}\n".format(path.with_suffix(".jpg").name, class_name))

    print("Setup done.")


if __name__ == "__main__":
    setup_example()
