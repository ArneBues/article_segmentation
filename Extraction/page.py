from PIL import Image
import config
from Training.SNN.patchHelper import PatchHelper
from Training.SNN.pageHandler import PageHandler


class Page(object):
    def __init__(self, page):
        self.file = page
        self.pageHandler = PageHandler(self.patchHelper.patch_count)

    @property
    def image(self):
        return Image.open(config.IMAGE_PATH + self.file)

    @property
    def patchHelper(self):
        return PatchHelper(self.file, config.SUPERPIXEL_SIZE, config.PATCH_AMOUNT)

    @property
    def image_size(self):
        return (self.image.size[1], self.image.size[0])
