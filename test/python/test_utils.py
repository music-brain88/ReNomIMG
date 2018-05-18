import pytest
from renom_img.api.utils.nms import nms
from renom_img.api.utils.load import parse_xml_detection
from renom_img.api.utils.target import build_target_yolo


@pytest.mark.parametrize("box_list, answer, threashold, return_type", [
    [[(10, 10, 20, 20), (5, 5, 20, 20)], [(5, 5, 20, 20)], 0.2, 'box'],
    [[(10, 10, 20, 20), (5, 5, 20, 20)], [1], 0.2, 'index']

])
def test_nms(box_list, answer, threashold, return_type):
    result = nms(box_list, threashold, return_type)
    assert result == answer

def test_parse_xml_detection():
    path = "voc.xml"
    annotation_list = parse_xml_detection([path])
    print("Parsed", annotation_list)
    assert annotation_list[0][0]['name']=='car'
    assert annotation_list[0][0]['box']==[43.5, 307.0, 61.0, 58.0]

def test_build_target_yolo():
    path = "voc.xml"
    annotation_list = parse_xml_detection([path])
    target, mapping = build_target_yolo(annotation_list, cells=4, img_size=(500, 500)) 
    print(target, mapping)
