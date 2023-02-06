import xml.etree.ElementTree as ET
from shapely import geometry as SH


class LabelHelper():

    def __init__(self, path, resize=1):
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.annotations = []
        self.testData = []
        self.resize = resize

        self.parseLabels()
        self.parseAnnotations()

    def parseLabels(self):
        self.labels = []
        labels = self.root[1][0][11]
        for label in labels:
            self.labels.append(label[0].text)

    def parseAnnotations(self):
        images = self.root.findall('image')
        for image in images:
            ann_image = {}
            ann_image['file'] = image.attrib['name']
            ann_image['id'] = image.attrib['id']
            ann_image['dimension'] = (
                int(image.attrib['width']), int(image.attrib['height']))
            ann_image['annotations'] = []

            for labelImage in image:
                label = {}
                label['class'] = labelImage.attrib['label']
                if(labelImage.tag == 'box'):
                    x1 = float(labelImage.attrib['xtl']) * self.resize
                    y1 = float(labelImage.attrib['ytl']) * self.resize
                    x2 = float(labelImage.attrib['xbr']) * self.resize
                    y2 = float(labelImage.attrib['ybr']) * self.resize

                    label['box'] = SH.Polygon(
                        [(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
                    label['type'] = 'box'

                elif(labelImage.tag == 'polygon'):
                    label['box'] = SH.Polygon(list(map(lambda x: [float(
                        y)*self.resize for y in x.split(',')], labelImage.attrib['points'].split(';'))))
                    label['type'] = 'polygon'

                ann_image['annotations'].append(label)

            if(len(ann_image['annotations']) > 0):
                self.annotations.append(ann_image)
            else:
                self.testData.append(ann_image)

    def find(self, imageName):
        for image in self.annotations:
            if image['file'] == imageName:
                return image
        return None
