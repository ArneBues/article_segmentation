IMAGE_PATH = "G:\Meine Ablage\Master\Bilder_small"
DEVICE = "cuda"

# SNN
ANNOTATION_FILE_SNN = "./labels/train_cluster.xml"
COMBINATION_AMOUNT = 500
BATCH_SIZE_SNN = 50
EPOCHES_SNN = 5
SUPERPIXEL_SIZE = (50, 50)
PATCH_RADIUS = 3

# FCN
ANNOTATION_FILE_FCN = "./labels/train_mask.xml"
BATCH_SIZE_FCN = 10
EPOCHES_FCN = 50
MASK_SIZE = (300, 204)

IMAGE_PATH = "./pages/"
CLUSTERING_MODEL = "../models/siamese.pth"
MASK_MODEL = "../models/mask.pth"

CLUSTERING_THRESHOLD = 0.8  # Higher Number = Greedier Clustering

PATCH_AMOUNT = 3
PATCH_SIZE = (SUPERPIXEL_SIZE[0] * (1 + 2 * PATCH_AMOUNT),
              SUPERPIXEL_SIZE[1] * (1 + 2 * PATCH_AMOUNT))

CLUSTERING_MASK_SIZE = (300, 204)
