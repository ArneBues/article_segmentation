import math
import PIL
import numpy as np
from sklearn import neighbors
from scipy.spatial import distance
from skimage.transform import rescale
import config
from shapely import geometry


class PageHandler:
    def __init__(self, dimension):
        self.dimension = dimension
        self.pixels = []
        self.segments = []
        self.sides_offsets = [
            (-1, 0),
            (0, -1),
            (0, 1),
            (1, 0),
        ]
        self.corner_offsets = [
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        self.matrix = []

    def add_image(self, image):
        self.image = PIL.Image.open(config.IMAGE_PATH + "/" + image)
        self.image_size = (self.image.size[1], self.image.size[0])

    def add_pixel(self, pixel):
        self.pixels.append(pixel)

    def create_matrix(self):
        self.matrix = [[0]*self.dimension[0] for i in range(self.dimension[1])]
        output = [[0]*self.dimension[0] for i in range(self.dimension[1])]

        for pixel in self.pixels:
            output[pixel.index[1]][pixel.index[0]] = pixel.value
            self.matrix[pixel.index[1]][pixel.index[0]] = pixel

        output = np.array(output)
        return output

    def get_cluster(self):
        output = []
        for y in self.matrix:
            for x in y:
                output.append(x.cluster)

        return output

    def add_clustering(self, clustering):
        y_count = 0
        for y in clustering:
            x_count = 0
            for x in y:
                self.matrix[y_count][x_count].add_cluster(x)
                x_count = x_count + 1

            y_count = y_count + 1

    def check_condition_neighbours(self, pixel, neighbour_count):
        index = pixel.index
        cluster = pixel.cluster

        neighbours = {}
        sides = 0
        corners = 0

        count = 0

        for offset in self.sides_offsets:
            neigbour = self.get_pixel_by_offset(index, offset)

            if neigbour is not None:
                if neigbour.cluster == cluster:
                    sides += 1

                if neigbour.cluster not in neighbours:
                    neighbours[neigbour.cluster] = 0
                neighbours[neigbour.cluster] += 1

        for offset in self.corner_offsets:
            neigbour = self.get_pixel_by_offset(index, offset)

            if neigbour is not None:
                if neigbour.cluster == cluster:
                    corners += 1
                if neigbour.cluster not in neighbours:
                    neighbours[neigbour.cluster] = 0
                neighbours[neigbour.cluster] += 1
        return ((sides, corners), neighbours)

    def get_pixel_by_offset(self, start_index, offset):
        x = start_index[0] + offset[0]
        y = start_index[1] + offset[1]

        if y > len(self.matrix) - 1:
            return None

        if x > len(self.matrix[y]) - 1:
            return None

        return self.matrix[y][x]

    def remove_wrong_pixels(self):
        for pixel in self.pixels:
            same_cluster, neighbors = self.check_condition_neighbours(pixel, 3)
            if(same_cluster[0] < 2 or same_cluster[1] < 1):
                pixel.cluster = max(neighbors, key=neighbors.get)

    def reassign_wrong_pixels(self):
        for pixel in self.pixels:
            if pixel.cluster == -1:
                neighbours = self.find_nearest_neighbours(pixel)

                min = -1
                nearest = None
                for neighbour in neighbours:
                    dist = distance.euclidean(neighbour.value, pixel.value)

                    if nearest is None or dist < min:
                        nearest = neighbour
                        min = dist

                pixel.better_cluster = nearest.cluster

    def finalize(self):
        for pixel in self.pixels:
            if pixel.better_cluster != -1:
                pixel.cluster = pixel.better_cluster

    def find_nearest_neighbours(self, pixel):

        offsets = np.array(self.sides_offsets)

        done = False
        attempts = 5
        neighbours = []
        i = 0
        while not done:
            for offset in offsets:
                neighbour = self.get_pixel_by_offset(pixel.index, offset*i)
                if neighbour is not None and neighbour.cluster != -1:
                    neighbours.append(neighbour)

            if len(neighbours) > 0 and i >= attempts:
                done = True
            else:
                i += 1

        return neighbours

    def find_edges(self):
        self.edges = [[[0]*2 for j in range(self.dimension[0])]
                      for i in range(self.dimension[1])]
        for y in self.matrix:
            for x in y:
                right = self.get_pixel_by_offset(x.index, (1, 0))
                bottom = self.get_pixel_by_offset(x.index, (0, 1))
                if right is not None and x.cluster != right.cluster:
                    self.fix_horizontal_edge(x)
                if bottom is not None and x.cluster != bottom.cluster:
                    pass
                    # self.fix_vertical_edge(x)
        print(self.edges)
        return self.edges

    def fix_horizontal_edge(self, pixel):
        a = self.get_pixel_by_offset(pixel.index, (-1, 0))
        b = pixel
        c = self.get_pixel_by_offset(pixel.index, (1, 0))
        d = self.get_pixel_by_offset(pixel.index, (2, 0))

        if None in (a, b, c, d):
            print("skipped")
            return

        bd = distance.euclidean(b.value, d.value)
        ac = distance.euclidean(a.value, c.value)
        if ac > bd:
            x = bd/ac
            pixel_index = b.index
        else:
            x = 1-ac/bd
            pixel_index = c.index

        self.edges[pixel_index[1]][pixel_index[0]][0] = x

    def fix_vertical_edge(self, pixel):
        a = self.get_pixel_by_offset(pixel.index, (0, -1))
        b = pixel
        c = self.get_pixel_by_offset(pixel.index, (0, 1))
        d = self.get_pixel_by_offset(pixel.index, (0, 2))

        if None in (a, b, c, d):
            print("skipped")
            return

        bd = distance.euclidean(b.value, d.value)
        ac = distance.euclidean(a.value, c.value)
        if ac > bd:
            x = 1-bd/ac
            pixel_index = b.index
        else:
            x = ac/bd
            pixel_index = c.index
        self.edges[pixel_index[1]][pixel_index[0]][1] = x

    def add_mask(self, mask):
        self.mask = mask

    def add_mask_segment(self, segment):
        self.segment.append(segment)

    def get_clusters_for_segment(self, segment):
        poly = geometry.Polygon(segment)
        clusters = []
        for pixel in self.pixels:
            if pixel.location.intersects(poly):
                clusters.append(pixel.cluster)
        return clusters

    def add_final_segments(self, segments):
        self.final_segments = segments

    def cluster_to_mask(self, shape, superPixelSize):
        output = [[0]*self.dimension[0] for i in range(self.dimension[1])]

        mask_dimension = (
            self.dimension[1]*superPixelSize[1], self.dimension[0]*superPixelSize[0])
        offset = (math.ceil((shape[1] - mask_dimension[1])/2),
                  math.ceil((shape[0] - mask_dimension[0])/2))

        for pixel in self.pixels:
            output[pixel.index[1]][pixel.index[0]] = pixel.cluster

        mask = np.negative(np.ones(shape))
        for yp in range(len(output)):
            y = yp * superPixelSize[0]
            for p in range(len(output[yp])):
                x = p * superPixelSize[1]

                for i in range(superPixelSize[0]):
                    for j in range(superPixelSize[1]):
                        mask[y + i + offset[1]][x + j +
                                                offset[0]] = output[yp][p]

        return mask


class Pixel:
    def __init__(self, index, value, location):
        self.index = index
        self.cluster = 0
        self.value = value
        self.better_cluster = -1
        self.location = location

    def add_cluster(self, cluster):
        self.cluster = cluster
