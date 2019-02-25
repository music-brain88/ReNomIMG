import numpy as np
from PIL import Image

mean = [103.939, 116.779, 123.68]


def load_img(img_path, size):
    img = Image.open(img_path)
    img = img.resize(size)
    return img


def preprocess_img(img):
    x = np.asarray(img, dtype='float32')
    x = x.transpose((2,0,1))
    x = x[np.newaxis,...]

    x = x[:, ::-1, :, :]
    x[:, 0, :, :] -= mean[0]  # R
    x[:, 1, :, :] -= mean[1]  # G
    x[:, 2, :, :] -= mean[2]  # B
    return x


def visualize(img, input_map, L, result, figsize=(16,16)):
    fig = plt.figure(figsize = (16,16)) 

    ax1 = fig.add_subplot(141)
    ax1.axis("off")
    ax1.imshow(img)

    ax2 = fig.add_subplot(142)
    ax2.axis("off")
    ax2.imshow(input_map)

    ax3 = fig.add_subplot(143)
    ax3.axis("off")
    ax3.imshow(L, cmap=cm.jet)

    ax4 = fig.add_subplot(144)
    ax4.axis("off")
    ax4.imshow(result)


def visualize_comparison(img, result_cam, result_cam_pp, L_cam, L_cam_pp, figsize=(12,12)):
    fig = plt.figure(figsize = figsize) 

    ax1 = fig.add_subplot(151)
    ax1.axis("off")
    ax1.imshow(img)

    ax2 = fig.add_subplot(152)
    ax2.axis("off")
    ax2.imshow(result_cam)

    ax3 = fig.add_subplot(153)
    ax3.axis("off")
    ax3.imshow(result_cam_pp)

    ax4 = fig.add_subplot(154)
    ax4.axis("off")
    ax4.imshow(L_cam, cmap=cm.jet)

    ax5 = fig.add_subplot(155)
    ax5.axis("off")
    ax5.imshow(L_cam_pp, cmap=cm.jet)
    plt.tight_layout()
