import numpy as np
import skimage.transform as transform
import skimage
import scipy.ndimage as nd
import colorsys
from PIL import Image
import config
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import math
import torch
import scipy.ndimage as nd
from skimage import segmentation

from skimage.draw import polygon
from skimage import measure
import numpy as np
import matplotlib.patches as mpatches


def mask_threshold(mask, threshold):
    return np.where(mask > threshold, 1, 0)


def resize(mask, size):
    return transform.resize(mask, size)


def mask_dilate(mask, iterations):
    return nd.binary_dilation(mask, iterations=iterations)


def mask_erode(mask, iterations):
    return nd.binary_erosion(mask, iterations=iterations)


def cluster(metrics, threshold):
    clustering = AgglomerativeClustering(affinity="euclidean", linkage="average",
                                         n_clusters=None, distance_threshold=threshold).fit(metrics)

    return np.array(clustering.labels_)


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


def run_mask_network(page, cnn, transform):
    image = page.image
    image = transform(image)
    image = torch.unsqueeze(image, 0)

    if config.DEVICE == "cuda":
        image = image.cuda()

    output1 = cnn(image)
    mask = output1[0].cpu().detach().numpy()[0]

    return (mask, image[0].cpu().detach().numpy())


def test(patchHelper, page, clustering, mask):
    im = patchHelper.image
    color = get_N_HexCol(max(clustering) + 1)
    fig, ax = plt.subplots(figsize=(6, 6))

    x = 0
    y = 0
    widthCount, heightCount = patchHelper.patch_count
    if mask is None:
        ax.imshow(im)
    else:
        im2 = np.array(im)
        mask = resize(mask, (im.size[1], im.size[0]))
        im2[mask == 0, :] = 0
        ax.imshow(im2)
    for i in range((widthCount * heightCount)):
        if(x > widthCount - 1):
            y = y + 1
            x = 0

        pixel = patchHelper.generateSuperPixelBox(i)

        bbox = mpatches.Rectangle(pixel[0], pixel[1][0], pixel[1][1],
                                  linewidth=0.5, color=color[math.floor(clustering[i])], alpha=0.7)
        #plt.text(pixel[0][0]+200, pixel[0][1]+200, str(x) + " " + str(y), fontsize=2)

        plt.text(pixel[0][0] + 20, pixel[0][1] + 20,
                 str(clustering[i]), fontsize=5, color='blue')
        ax.add_patch(bbox)

        x = x + 1
    plt.rcParams['figure.dpi'] = 500
    plt.axis('off')
    plt.show()


def fix_mask(mask, shape, erode=3, dilate=2):
    mask = mask_threshold(mask, 0.96)
    mask = nd.binary_fill_holes(mask)
    mask = mask_erode(mask, erode)
    mask = mask_dilate(mask, dilate)
    mask = nd.binary_fill_holes(mask)
    mask = resize(mask, shape)
    return mask


def combine_cluster_mask(page, mask, expand=True):

    contours = findContours(mask)

    fullMask = np.zeros(mask.shape)
    for contour in contours:
        if len(contour) > 100:
            mask2 = np.zeros(page.image_size)
            rr, cc = polygon(contour[:, 0], contour[:, 1])
            mask2[rr, cc] = 1
            mask2 = nd.binary_fill_holes(mask2)
            if (mask[rr, cc] == True)[0]:
                clusters = page.pageHandler.get_clusters_for_segment(
                    list(zip(cc, rr)))
                cluster = max(clusters, key=clusters.count)

                test = np.array(mask2, dtype=int)
                test[test == 1] = cluster + 1

                fullMask = fullMask + test
    if expand:
        fullMask = segmentation.expand_labels(fullMask, distance=1000)
        #fullMask = measure.label(fullMask)
    return fullMask
