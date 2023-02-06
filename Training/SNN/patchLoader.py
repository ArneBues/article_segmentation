from Training.SNN.labelHelper import LabelHelper
from Training.SNN.patchHelper import PatchHelper
import random
import config


class PatchLoader():

    def __init__(self, labelHelper: LabelHelper, shuffle: bool = False, superPixelSize=[256, 256], patchSize=2, testData=False, testCount=50):

        self.labelHelper = labelHelper
        self.shuffle = shuffle
        self.superPixelSize = superPixelSize
        self.patchSize = patchSize

        if not testData:
            self.data = random.sample(
                labelHelper.annotations, len(labelHelper.annotations))
        else:
            self.data = labelHelper.testData[-testCount:]

        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.data):
            raise StopIteration
        out = self.data[self.current]
        self.current = self.current + 1

        patchHelper = self.initPatchHelper(out['file'])

        superPixelsWithLabel = self.assignLabelToPixel(patchHelper, out)

        combinations = self.createCombinations(
            superPixelsWithLabel, patchHelper)

        return {
            'data': out,
            'id': out.get('id'),
            'combinations': combinations,
            'patchHelper': patchHelper,
            'superPixelsWithLabel': superPixelsWithLabel
        }

    def initPatchHelper(self, file):
        patchHelper = PatchHelper(file, self.superPixelSize, self.patchSize)
        return patchHelper

    def assignLabelToPixel(self, patchHelper, data):
        superPixelsWithLabel = {}

        for y in range(len(patchHelper.super_pixels)):
            for x in range(len(patchHelper.super_pixels[y])):
                total = []
                text = ''
                pixel = ''
                for ann in data['annotations']:
                    pixel = patchHelper.super_pixels[y][x]['location']
                    intersection = ann['box'].intersection(pixel).area

                    percentage = intersection / pixel.area

                    total.append(
                        {"class": ann['class'], "percentage": percentage})

                max = {"class": 'b', "percentage": 0}
                sum = 0
                for i in range(len(total)):
                    sum = sum + total[i]['percentage']
                    if max['percentage'] < total[i]['percentage']:
                        max = total[i]

                # if (1 - sum > max['percentage']):
                #    max = {"class": 'b', "percentage": 1 - sum}

                max['index'] = (x, y)

                if max['class'] in superPixelsWithLabel:
                    superPixelsWithLabel[max['class']].append(max)
                else:
                    superPixelsWithLabel[max['class']] = []
                    superPixelsWithLabel[max['class']].append(max)

        return superPixelsWithLabel

    def createCombinations(self, superPixelsWithLabel, patchHelper):
        correct = []
        incorrect = []
        pixels = []
        for i in superPixelsWithLabel:
            pixels = pixels + superPixelsWithLabel[i]

        for i in range(len(pixels)):
            for j in range(i + 1, len(pixels)):
                if pixels[i]['class'] == pixels[j]['class']:
                    correct.append((pixels[i], pixels[j], patchHelper))
                else:
                    incorrect.append((pixels[i], pixels[j], patchHelper))

        if len(correct) > config.COMBINATION_AMOUNT:
            correct = random.sample(correct, config.COMBINATION_AMOUNT)

        if len(incorrect) > len(correct):
            incorrect = random.sample(incorrect, len(correct))

        output = (incorrect + correct)
        random.shuffle(output)

        return output
