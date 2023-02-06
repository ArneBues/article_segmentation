import xml.etree.ElementTree as ET
from skimage.draw import polygon, rectangle
import math
import numpy as np


class Label:
    def __init__(self, path, resize=1, size=(881, 598)):
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.annotations = []
        self.testData = []
        self.size = size
        self.resize_factor = resize
        self.parseLabels()
        self.parseAnnotations()

    def parseLabels(self):
        self.labels = []
        labels = self.root[1][0][11]
        for label in labels:
            self.labels.append(label[0].text)

    def findFile(self, file):
        for a in self.testData:
            if a['file'] == file:
                return a
        for image in self.annotations:
            if image['file'] == file:
                return image
        return None

    def parseAnnotations(self):
        images = self.root.findall('image')
        for image in images:
            ann_image = {}
            ann_image['file'] = image.attrib['name']
            ann_image['id'] = image.attrib['id']
            ann_image['dimension'] = (
                int(image.attrib['width']), int(image.attrib['height']))

            if self.size is not None:
                self.resize = (self.size[0] / ann_image['dimension'][0],
                               self.size[1] / ann_image['dimension'][1])
            else:
                self.resize = np.array((1, 1)) * self.resize_factor
            ann_image['annotations'] = []
            ann_image['mask'] = np.zeros(
                (math.ceil(ann_image['dimension'][1]*self.resize[1]), math.ceil(ann_image['dimension'][0]*self.resize[0])))

            ann_image['dimension'] = (
                int(ann_image['dimension'][0]*self.resize[0]), int(ann_image['dimension'][1]*self.resize[1]))

            for labelImage in image:
                label = {}
                label['class'] = labelImage.attrib['label']
                if(labelImage.tag == 'box'):
                    x1 = math.ceil(
                        float(labelImage.attrib['xtl']) * self.resize[0])
                    y1 = math.ceil(
                        float(labelImage.attrib['ytl']) * self.resize[1])
                    x2 = math.ceil(
                        float(labelImage.attrib['xbr']) * self.resize[0])
                    y2 = math.ceil(
                        float(labelImage.attrib['ybr']) * self.resize[1])

                    label['box'] = rectangle((y1, x1), (y2, x2))
                    label['type'] = 'box'

                elif(labelImage.tag == 'polygon'):
                    row = ()
                    col = ()
                    data = labelImage.attrib['points'].split(';')

                    for point in data:
                        x = math.ceil(
                            float(point.split(',')[0]) * self.resize[0])
                        y = math.ceil(
                            float(point.split(',')[1]) * self.resize[1])
                        row = row + (y,)
                        col = col + (x,)
                    label['box'] = polygon(row, col)
                    label['type'] = 'polygon'

                rr, cc = label['box']
                ann_image['mask'][rr-1, cc-1] = 1
                ann_image['annotations'].append(label)

            if(len(ann_image['annotations']) > 0):
                self.annotations.append(ann_image)
            else:
                self.testData.append(ann_image)
