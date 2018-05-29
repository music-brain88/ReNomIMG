import numpy as np
import renom_img.api.utils.nms import 

cpdef calc_mAP(predicted, target):
    """
    """
    pass

cpdef calc_IOU(predicted, target):
    """
    Format of predicted.
    {
      [ # Objects of 1 st image.
        {'box': [x(float), y, w, h], 'name': class_name(string)},
        {'box': [x(float), y, w, h], 'name': class_name(string)}
      ],
      [ # Objects of 2nd image.
        {'box': [x(float), y, w, h], 'name': class_name(string)},
        {'box': [x(float), y, w, h], 'name': class_name(string)}
      ]
    }
    The function 'renom_img.api.utils.prediction.build_'

    Format of target.
    {
      [ # Objects of 1 st image.
        {'box': [x(float), y, w, h], 'name': class_name(string)},
        {'box': [x(float), y, w, h], 'name': class_name(string)}
      ],
      [ # Objects of 2nd image.
        {'box': [x(float), y, w, h], 'name': class_name(string)},
        {'box': [x(float), y, w, h], 'name': class_name(string)}
      ]
    }

    The function `renom_img.api.utils.load.parse_xml_detection` returns 
    this formatted dictionary.
    """
