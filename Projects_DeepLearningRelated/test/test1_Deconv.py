# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torch import nn


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return img


class ToTensor(object):
    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, img):
        img = img.resize(self.size, Image.BILINEAR)
        return img


def transform_tr(img):
    composed_transforms = transforms.Compose([
        FixedResize(512),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])
    return composed_transforms(img)


class TestModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        score = self.bn1(img)
        img1 = self.deconv1(score)
        score = self.relu(img1)
        return score

if __name__ == "__main__":
    torch.cuda.get_device_name(0)
    # _image = Image.open("90.JPG").convert("RGB")
    # _imagepro = transform_tr(_image)
    # testm=TestModule().cuda()
    # ret=testm(_imagepro)
    pass
