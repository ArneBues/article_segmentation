import config
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from Training.SNN.model import Network, NetworkResnet, DataLoaderClustering, DataLoaderClusteringExtraction
import torch.nn as nn
import torchvision.transforms as transforms
import torch
from Training.SNN.pageHandler import PageHandler, Pixel


def cluster_superpixels(matrix):
    shape = matrix.shape
    pixels = matrix.reshape(shape[0]*shape[1], shape[2])

    clustering = AgglomerativeClustering(affinity="euclidean", linkage="average",
                                         n_clusters=None, distance_threshold=config.CLUSTERING_THRESHOLD).fit(pixels)

    matrix = np.reshape(np.array(clustering.labels_), (shape[0], shape[1]))

    return matrix


def cluster(metrics, threshold):
    clustering = AgglomerativeClustering(affinity="euclidean", linkage="average",
                                         n_clusters=None, distance_threshold=threshold).fit(metrics)

    return np.array(clustering.labels_)


def run_clustering(page, patches, cnn, convert_tensor):
    page_data = {
        'patchHelper': page.patchHelper,
        'patches': flatten(patches),
    }

    dataset = DataLoaderClusteringExtraction(page_data, convert_tensor)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=15,
                                         shuffle=False)
    for img, index, mask in loader:
        if config.DEVICE == 'cuda':
            img = img.cuda()
            mask = mask.cuda()
        output = cnn.sub_forward(img, mask)

        for o in range(len(output)):
            page.pageHandler.add_pixel(Pixel(((index[0][o], index[1][o])), output.cpu()[o].detach(
            ).numpy(), page.patchHelper.getPixelPolygon(index[0][o], index[1][o])))
        # print(index)

    clusters = cluster_superpixels(page.pageHandler.create_matrix())
    return clusters


def flatten(t):
    return [item for sublist in t for item in sublist]
