import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys

default_color_list = [
    (176, 58, 46, 150),
    (91, 44, 111, 150),
    (40, 116, 166, 150),
    (17, 122, 101, 150),
    (35, 155, 86, 150),
    (185, 119, 14, 150),
    (160, 64, 0, 150),
    (144, 148, 151, 150),
    (97, 106, 107, 150),
    (33, 47, 61, 150),
    (205, 97, 85, 150),
    (175, 122, 197, 150),
    (84, 153, 199, 150),
    (72, 201, 176, 150),
    (82, 190, 128, 150),
    (244, 208, 63, 150),
    (245, 176, 65, 150),
    (235, 152, 78, 150),
    (240, 243, 244, 150),
    (149, 165, 166, 150),
    (52, 73, 94, 150),
]


def draw_box(img, prediction, font_path=None, color_list=None):
    """Function for describing bounding box, class name and socre for an input image.

    Example:
        >>> from PIL import Image
        >>> from renom_img.api.utility.load import *
        >>> prediction = load(prediction_xml_path)
        >>> image = Image.open(img_path)
        >>> bbox_image = draw_bbox(img_path, prediction)

    Args:
        img(string):
        prediction(list): List of annotations.
            Each annotation has a list of dictionary which includes keys 'box', 'name' and 'score'.
            The format is below.

    .. code-block :: python

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
    if color_list is None:
        color_list = default_color_list

    if isinstance(img, str):
        img = Image.open(img).convert("RGBA")
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8)).convert("RGBA")

    w, h = img.size
    canvas = Image.new("RGBA", (w, h), "#00000000")
    draw = ImageDraw.Draw(canvas)

    for params in prediction:
        box = params['box']
        class_id = params.get('class', 0) % len(color_list)
        name = params.get('name', None)
        score = params.get('score', None)

        x1 = (box[0] - box[2] / 2.) * w
        y1 = (box[1] - box[3] / 2.) * h
        x2 = (box[0] + box[2] / 2.) * w
        y2 = (box[1] + box[3] / 2.) * h
        for i in range(-2, 3):
            for j in range(-2, 3):
                draw.rectangle([x1 + i, y1 + j, x2 + i, y2 + j], outline=color_list[class_id])

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
                (x1 - 2, y1, x1 + text_size[0] + 5 - 2, y1 + text_size[1]), fill=color_list[class_id])
            draw.text((x1 + 5 - 2, y1 - 1), text, (255, 255, 255, 250), font=font)
    return Image.alpha_composite(img, canvas)


def draw_segment(img, prediction, font_path=None, color_list=None):
    """

    Example:
        >>> from PIL import Image
        >>> from renom_img.api.utility.load import *
        >>> prediction = load(prediction_xml_path)
        >>> image = Image.open(img_path)
        >>> segment_image = draw_segment(img_path, prediction)

    Args:
        img(str): path for target image
        prediction(list): List of annotations.
            Each annotation has a list of dictionary which includes keys 'box', 'name' and 'score'.
            The format is below.

    .. code-block :: python

        [
            {'box': [x(float), y, w, h], 'name': class name(string), 'score': score(float)},
            {'box': [x(float), y, w, h], 'name': class name(string), 'score': score(float)},
            ...
        ]

        font_path: font path of chanracters on the image

    Return:
        (PIL.Image): This returns image described segment.
    """
    if color_list is None:
        color_list = default_color_list

    img = Image.open(img).convert("RGBA")
    h, w = prediction.shape
    canvas = Image.new("RGBA", (w, h), "#00000000")

    class_num = np.max(prediction) - np.min(prediction)

    for c in range(class_num):
        mask = Image.fromarray(np.where(prediction == c, True, False)).convert("L")
        class_canvas = Image.new("RGBA", (w, h), color_list[c % len(color_list)])
        canvas.paste(class_canvas, mask=mask)

    return Image.alpha_composite(img, canvas.resize(img.size, Image.BILINEAR))


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
