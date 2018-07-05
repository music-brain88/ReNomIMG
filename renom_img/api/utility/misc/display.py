import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys


def draw_box(img_path, prediction_list, font_path=None, color=(0, 0, 255, 150)):
    """Function for describing bounding box, class name and socre for an input image.

    Example:
        >>> from PIL import Image
        >>> from renom_img.api.utility.load import *
        >>> prediction_list = load(prediction_xml_path)
        >>> image = Image.open(img_path)
        >>> bbox_image = draw_bbox(img_path, prediction_list)

    Args:
        img_path(string):
        prediction_list(list): List of annotations.
            Each annotation has a list of dictionary which includes keys 'box', 'name' and 'score'.
            The format is below.

        [
            {'box': [x(float), y, w, h], 'name': class name(string), 'score': score(float)},
            {'box': [x(float), y, w, h], 'name': class name(string), 'score': score(float)},
            ...
        ]
        font_path(string):

    Returns:
        (PIL.Image): This returns image described prediction result.

    Note:
        The values of `box` is a relational coordinate so their values are in [0.0 ~ 1.0].
    """
    img = Image.open(img_path).convert("RGBA")
    w, h = img.size
    canvas = Image.new("RGBA", (w, h), "#00000000")
    draw = ImageDraw.Draw(canvas)
    
    for params in prediction_list:
        box = params['box']
        name = params.get('name', None)
        score = params.get('score', None)

        x1 = (box[0] - box[2] / 2.) * w
        y1 = (box[1] - box[3] / 2.) * h
        x2 = (box[0] + box[2] / 2.) * w
        y2 = (box[1] + box[3] / 2.) * h
        for i in range(-2, 3):
            for j in range(-2, 3):
                draw.rectangle([x1 + i, y1 + j, x2 + i, y2 + j], outline=color)

        text = None
        if name and score:
            text = "{}:{:.1f}%".format(name, score * 100)
        elif name:
            text = "{}".format(name)
        elif score:
            text = "{:.1f}%".format(score * 100)

        if font_path is not None:
            font_path = font_path
        else:
            font_path = os.path.join(os.path.dirname(__file__), "FreeSansBold.ttf")

        fontsize = 30
        font = ImageFont.truetype(font_path, int(fontsize))
        if text is not None:
            text_size = font.getsize(text)
            draw.rectangle(
                (x1 - 2, y1, x1 + text_size[0] + 5 - 2, y1 + text_size[1]), fill=color)
            draw.text((x1 + 5 - 2, y1 - 1), text, (255, 255, 255, 250), font=font)
    return Image.alpha_composite(img, canvas)


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
