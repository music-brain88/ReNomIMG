from PIL import Image, ImageDraw, ImageFont


def draw_box(img, bbox, class_name=None, score=None):
    """
    Args:
        img: PIL Image object.
        bbox(list): List of coordinate. [x, y, w, h]

    """
    draw = ImageDraw.Draw(img)
    for j in range(-2, 3):
        x1 = bbox[0] - bbox[2] / 2.
        y1 = bbox[1] - bbox[3] / 2.
        x2 = bbox[0] + bbox[2] / 2.
        y2 = bbox[1] + bbox[3] / 2.
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0))


def pil2array(img):
    """
    """
    return np.asarray(img).transform(2, 0, 1).copy()
