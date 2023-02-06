from PIL import Image
import math
from shapely import geometry as SH
import config
import warnings
import numpy as np


class PatchHelper():

    def __init__(self, image: str, sp_dim=[256, 256], rel_patch_dim=2):
        self.sp_dim = sp_dim
        self.rel_patch_dim = rel_patch_dim
        self.patch_dim = (
            sp_dim[0] * rel_patch_dim,
            sp_dim[1] * rel_patch_dim
        )
        self.rel_patch_dim = rel_patch_dim
        self.file = image
        self.cropped = False
        self.patches = []
        self.loadImage()
        self.im_size = self.image.size

        self.patch_count = self.calculate_patch_amount()
        self.offset = self.calculate_offset()
        self.generateSuperPixels()
        # self.generateAllPatches()

        self.smallImage = self.getSmallImage()

    def loadImage(self):
        self.image = Image.open(config.IMAGE_PATH + "/" + self.file)

    def getSmallImage(self):
        return self.image.resize(config.MASK_SIZE)

    def createMask(self, xP, yP):
        scale = self.calculate_scale()

        mask = np.zeros((config.MASK_SIZE[1], config.MASK_SIZE[0])) - 1

        pixel = self.getSuperPixel(xP, yP)
        x0, y0, x1, y1 = pixel['location'].bounds

        x0 = math.floor(x0 * scale[0])
        x1 = math.ceil(x1 * scale[0])
        y0 = math.floor(y0 * scale[1])
        y1 = math.ceil(y1 * scale[1])

        for y in range(len(mask)):
            for x in range(len(mask[y])):
                if y >= y0 and y <= y1 and x >= x0 and x <= x1:
                    mask[y][x] = 1

        return mask

    def calculate_scale(self):
        w0, h0 = self.im_size
        w1, h1 = config.MASK_SIZE

        return ((w1 / w0), (h1 / h0))

    def getSuperPixel(self, x, y):
        return self.super_pixels[y][x]

    def calculate_patch_amount(self):
        widthCount = math.floor(self.im_size[0] / self.sp_dim[0])
        heightCount = math.floor(self.im_size[1] / self.sp_dim[1])
        return (widthCount, heightCount)

    def calculate_offset(self):
        width_offset = math.floor(
            (self.im_size[0] - (self.patch_count[0] * self.sp_dim[0])) / 2)
        height_offset = math.floor(
            (self.im_size[1] - (self.patch_count[1] * self.sp_dim[1])) / 2)
        return (width_offset, height_offset)

    def generateSuperPixelBox(self, number):
        width = number % self.patch_count[0]
        height = math.floor(number/self.patch_count[0])
        point = (self.offset[0] + self.sp_dim[0] * (width),
                 self.offset[1] + self.sp_dim[1] * (height))
        dimension = (self.sp_dim[0], self.sp_dim[1])
        return (point, dimension)

    def generateSuperPixels(self):
        self.super_pixels = [[]]
        x = 0
        y = 0
        amount = self.patch_count[0] * self.patch_count[1]
        for i in range(amount):
            if(x > self.patch_count[0] - 1):
                y = y + 1
                x = 0
                self.super_pixels.append([])

            pixel = self.generateSuperPixelBox(i)
            result = pixel[0] + (pixel[1][0] + pixel[0][0],
                                 pixel[1][0] + pixel[0][1])

            self.super_pixels[y].append({
                'pixel': self.image.crop(pixel[0] + (pixel[1][0] + pixel[0][0], pixel[1][0] + pixel[0][1])),
                'location': SH.Polygon([(result[0], result[1]), (result[0], result[3]), (result[2], result[3]), (result[2], result[1])])
            })
            x = x + 1

    def generateAllPatches(self):
        for i in range(self.patch_count[1]):
            patches = []
            for j in range(self.patch_count[0]):
                patches.append(self.generatePatch(i, j))
            self.patches.append(patches)

    def generatePatch(self, superX, superY):
        new_im = Image.new('RGB', size=(
            self.sp_dim[0] * (self.rel_patch_dim * 2 + 1),
            self.sp_dim[1] * (self.rel_patch_dim * 2 + 1)))

        total = (self.rel_patch_dim * 2 + 1) * (self.rel_patch_dim * 2 + 1)

        offset_x = 0
        offset_y = 0
        x_start = superX - self.rel_patch_dim
        y_start = superY - self.rel_patch_dim
        x = x_start
        y = y_start

        for i in range(total):
            if x >= 0 and x < self.patch_count[0] and y >= 0 and y < self.patch_count[1]:
                new_im.paste(self.super_pixels[y][x]
                             ['pixel'], (offset_x, offset_y))
            offset_x = offset_x + self.sp_dim[0]
            x = x + 1
            if (x - x_start) > (self.rel_patch_dim * 2):
                x = x_start
                y = y + 1
                offset_y = offset_y + self.sp_dim[1]
                offset_x = 0

        return new_im

    def getPixelPolygon(self, x, y):
        return self.super_pixels[y][x]['location']

    def getPixelPolygonByIndex(self, index):
        y = math.floor(index / self.patch_count[0])
        x = index % self.patch_count[0]

        return self.getPixelPolygon(x, y)

    def generatePatchByIndex(self, index):
        y = math.floor(index / self.patch_count[0])
        x = index % self.patch_count[0]

        return (self.generatePatch(x, y), (x, y))

    def getPatch(self, x, y):
        return self.generatePatch(x, y)

    def getIndex(self, index):
        y = math.floor(index / self.patch_count[0])
        x = index % self.patch_count[0]

        return (x, y)

    def getPatchByIndex(self, index):
        y = math.floor(index / self.patch_count[0])
        x = index % self.patch_count[0]

        return (self.getPatch(x, y), (x, y))
