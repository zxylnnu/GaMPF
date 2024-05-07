import torch
import torch.nn as nn
from util import upsample
from GaMPF_element import *

class GaMPF(nn.Module):

    def __init__(self, encoder_arch):
        super(GaMPF, self).__init__()

        self.encoder1, channels = encoder(encoder_arch,pretrained=True)
        self.encoder2, _ = encoder(encoder_arch,pretrained=True)
        self.FsG = FsG(channels)
        self.classifier = nn.Conv2d(channels[0], 2, 1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.cls = nn.Sequential(
            nn.Conv2d(896, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

    def forward(self, img):
        img_t0,img_t1 = torch.split(img,3,1)
        features_t0 = self.encoder1(img_t0)
        features_t1 = self.encoder2(img_t1)
        features = features_t0 + features_t1
        features_map = self.FsG(features)

        pred_ = self.cls(features_map)
        pred_ = upsample(pred_,[pred_.size()[2]*2, pred_.size()[3]*2])
        pred_ = self.bn(pred_)
        pred_ = upsample(pred_,[pred_.size()[2]*2, pred_.size()[3]*2])
        pred_ = self.relu(pred_)
        pred = self.classifier(pred_)

        return pred
