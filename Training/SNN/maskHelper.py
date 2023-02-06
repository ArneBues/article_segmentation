import numpy as np
import skimage.transform as transform
import skimage
import scipy.ndimage as nd
import colorsys
from PIL import Image
import config
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as pltimage


def mask_threshold(mask, threshold):
    return np.where(mask > threshold, 1, 0)


def resize(mask, size):
    return transform.resize(mask, size)


def mask_dilate(mask, iterations):
    return nd.binary_dilation(mask, iterations=iterations)


def mask_erode(mask, iterations):
    return nd.binary_erosion(mask, iterations=iterations)


def get_N_HexCol(N=5):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out


def showImageWithMask(file, mask=None, maskColor=None, save=False):

    im = Image.open(config.IMAGE_PATH + "/" + file)

    matplotlib.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(figsize=(6, 6))
    if mask is None:
        ax.imshow(im)
    else:
        im2 = np.array(im)
        im2[mask == 0, :] = 0
        ax.imshow(im2)

    if maskColor is not None:
        plt.imshow(maskColor, cmap='jet', alpha=0.5)

    plt.axis('off')
    if save == False:
        plt.show()
    else:
        plt.savefig(save)


def findContours(mask):
    return skimage.measure.find_contours(mask)
