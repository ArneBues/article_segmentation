import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class Network(nn.Module):
    def __init__(self, img_size=[350, 350]):
        super().__init__()
        self.cnn_patch = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=384,
                      kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=512,
                      kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256,
                      kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten())

        self.cnn_mask = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64,
                      kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=384,
                      kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten())

        n_channels_patch = self.cnn_patch(torch.empty(
            1, 3, img_size[0], img_size[1])).size(-1)

        n_channels_mask = self.cnn_mask(torch.empty(1, 4, 204, 300)).size(-1)

        self.fc_mask = nn.Sequential(
            nn.Linear(n_channels_mask, 512)
        )

        self.fc_patch = nn.Sequential(
            nn.Linear(n_channels_patch, 512)
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def sub_forward(self, x, mask):
        patch = self.cnn_patch(x)
        mask = self.cnn_mask(mask)
        out_patch = self.fc_patch(patch)
        out_mask = self.fc_mask(mask)
        output = self.fc(torch.cat((out_patch, out_mask), 1))
        return output

    def forward(self, input1, index1, input2, index2):
        output1 = self.sub_forward(input1, index1)
        output2 = self.sub_forward(input2, index2)
        return output1, output2


class NetworkResnet(nn.Module):
    def __init__(self, img_size=[350, 350], train_all_layers=True):
        super().__init__()
        self.cnn_patch = torchvision.models.resnet50(pretrained=True)

        self.cnn_mask = torchvision.models.resnet50(pretrained=True)

        weight = self.cnn_mask.conv1.weight.clone()
        # here 4 indicates 4-channel input
        self.cnn_mask.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.cnn_mask.conv1.weight[:, :3] = weight
            self.cnn_mask.conv1.weight[:, 3] = self.cnn_mask.conv1.weight[:, 0]

        if not train_all_layers:
            for param in self.cnn_patch.parameters():
                param.requires_grad = False

            for param in self.cnn_mask.parameters():
                param.requires_grad = False

        self.cnn_patch.fc = nn.Linear(2048, 512)
        self.cnn_mask.fc = nn.Linear(2048, 512)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def sub_forward(self, x, mask):
        patch = self.cnn_patch(x)
        mask = self.cnn_mask(mask)
        output = self.fc(torch.cat((patch, mask), 1))
        return output

    def forward(self, input1, index1, input2, index2):
        output1 = self.sub_forward(input1, index1)
        output2 = self.sub_forward(input2, index2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forwardOld(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

    def forward(self, output1, output2, label):
        dist = output1 - output2
        EW = torch.sqrt(torch.sum(torch.pow(dist, 2), 1))
        Q = 2
        loss = (1 - label) * (2 / Q) * torch.pow(EW, 2) \
            + label * 2 * Q * torch.exp((-2.77/Q) * EW)

        return loss.mean()


class DataLoaderDynamic(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        convert_tensor = self.transform
        current = self.data[index]

        img1_data = current[0]
        img2_data = current[1]
        helper = current[2]

        if img1_data['class'] == img2_data['class']:
            label = 0
        else:
            label = 1

        img1 = convert_tensor(helper.getPatch(*img1_data['index']))
        img2 = convert_tensor(helper.getPatch(*img2_data['index']))

        img_small = convert_tensor(helper.smallImage)

        img1_mask = torch.tensor(helper.createMask(*img1_data['index']))
        img2_mask = torch.tensor(helper.createMask(*img2_data['index']))

        img1_masked = torch.cat((img_small, img1_mask[None]), 0).float()
        img2_masked = torch.cat((img_small, img2_mask[None]), 0).float()

        return (img1, img1_masked, img2, img2_masked, label, helper.file)


class DataLoaderClusteringExtraction(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transform = transforms

    def __len__(self):
        return len(self.data['patches'])

    def __getitem__(self, index):
        convert_tensor = self.transform

        helper = self.data['patchHelper']

        img, xandy = helper.generatePatchByIndex(index)

        img_small = convert_tensor(helper.smallImage)

        img1_mask = torch.tensor(helper.createMask(*xandy))
        img1_masked = torch.cat((img_small, img1_mask[None]), 0).float()

        img = convert_tensor(img)

        return (img, xandy, img1_masked)


class DataLoaderClustering(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transform = transforms

    def __len__(self):
        return len(self.data['pixels'])

    def __getitem__(self, index):
        convert_tensor = self.transform
        current = self.data['pixels'][index]

        label = current['class']
        helper = self.data['patchHelper']

        img, xandy = helper.generatePatchByIndex(index)

        img_small = convert_tensor(helper.smallImage)

        img1_mask = torch.tensor(helper.createMask(*xandy))
        img1_masked = torch.cat((img_small, img1_mask[None]), 0).float()

        img = convert_tensor(img)

        return (img, label, xandy, img1_masked)
