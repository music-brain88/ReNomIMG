import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys


def draw_box(img_path, prediction_list):
    """Function for describing bounding box, class name and socre for an input image.

    Example:
        >>> from PIL import Image
        >>> from renom_img.api.utility.load import *
        >>> prediction_list = load(prediction_xml_path)
        >>> image = Image.open(img_path)
        >>> bbox_image = draw_bbox(img_path, prediction_list)

    Args:
        prediction_list(list): List of annotations.
            Each annotation has a list of dictionary which includes keys 'box', 'name' and 'score'.
            The format is below.

        [
            {'box': [x(float), y, w, h], 'name': class name(string), 'score': score(float)},
            {'box': [x(float), y, w, h], 'name': class name(string), 'score': score(float)},
            ...
        ]

    Returns:
        (PIL.Image): This returns image described prediction result.
    """
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for params in prediction_list:
        box = params['box']
        name = params['name'] if 'name' in params else None
        score = params['score'] if 'score' in params else None

        x1 = box[0] - box[2] / 2.
        y1 = box[1] - box[3] / 2.
        x2 = box[0] + box[2] / 2.
        y2 = box[1] + box[3] / 2.
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0))

        text = None
        if name and score:
            text = "{}:{:.1f}%".format(name, score*100)
        elif name:
            text = "{}".format(name)
        elif score:
            text = "{:.1f}%".format(score*100)

        if sys.platform == "darwin": # OSX
            FONTPATH = '/Library/Fonts/Verdana.ttf'
        elif sys.platform == "linux2": # Linux
            FONTPATH = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'
        else: # Windows
            FONTPATH = 'C:\WINDOWS\Fonts\MSGOTHIC.ttc'

        fontsize = 10 if box[3] * 0.1 < 10 else box[3] * 0.1

        font = ImageFont.truetype(FONTPATH, int(fontsize))
        if text is not None:
            text_size = font.getsize(text)
            draw.rectangle((x1, y1-text_size[1], x1+text_size[0]*1.2, y1), fill=(255, 0, 0))
            draw.text((x1+5, y1-text_size[1]-1), text, (255, 255, 255), font=font)

    return img


def pil2array(img):
    """Function for convert PIL image to numpy array.

    Example:
         >>> from renom_img.api.utility.misc.display import pil2array
         >>> from PIL import Image
         >>> img = Image.open(img_path)
         >>> converted_img = pil2array(img)

    Args:
        img(PIL.Image): PIL Image

    Return:
        (numpy.ndarray): This returns numpy array object.

    """
    return np.asarray(img).transpose(2, 0, 1).copy()
