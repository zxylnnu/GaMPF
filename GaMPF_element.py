import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.utils.model_zoo as model_zoo
from util import *
import torch.nn.functional as F

__all__ = ['encoder', 'FsG']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def block_function_factory(conv, norm, relu=None):
    def block_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x

    return block_function

def do_efficient_fwd(block_f, x, efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block_f, x)
    else:
        return block_f(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1, downsample=None, efficient=True, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        block_f1 = block_function_factory(self.conv1, self.bn1, self.relu)
        block_f2 = block_function_factory(self.conv2, self.bn2)

        out = do_efficient_fwd(block_f1, x, self.efficient)
        out = do_efficient_fwd(block_f2, out, self.efficient)

        out = out + residual
        relu_out = self.relu(out)

        return relu_out, out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True):
        super(Bottleneck, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if self.use_bn else None
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = block_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = block_function_factory(self.conv2, self.bn2, self.relu)
        bn_3 = block_function_factory(self.conv3, self.bn3)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)
        out = do_efficient_fwd(bn_3, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu_out = self.relu(out)

        return relu_out, out

class ResNet(nn.Module):

    def __init__(self, block, layers, efficient=False, use_bn=True, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.use_bn = use_bn
        self.efficient = efficient

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward(self, image):

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [skip]
        return features

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    channels = [64, 128, 256, 512]
    return model, channels

class DenseBlock(nn.Sequential):
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm_layer=nn.BatchNorm2d):
        super(DenseBlock, self).__init__()
        if bn_start:
            self.add_module('norm1', norm_layer(input_num)),

        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', norm_layer(num1)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(DenseBlock, self).forward(_input)
        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)
        return feature

def conv_bn_relu(in_channels, out_channels, kernel_size=1, stride=1, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                  bias=False),
        norm_layer(out_channels),
        nn.ReLU(inplace=True)
    )

def conv_sigmoid(in_channels, out_channels, kernel_size=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
    )

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    # layer = getattr(cfg.MODEL, 'BNFUNC')
    layer = torch.nn.BatchNorm2d
    normalization_layer = layer(in_channels)
    return normalization_layer

def Up(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)

######################################################
def encoder(arch, pretrained=True):
    if arch == 'resnet18':
        return resnet18(pretrained)

def FsG(channels=[64, 128, 256, 512]):
    return FsGwithCE(channels=channels)

################################################
class FsGwithCE(nn.Module):

    def __init__(self, channels):
        super(FsGwithCE, self).__init__()

        self.downsample1 = conv3x3(channels[0], channels[1], stride=2)
        self.downsample2 = conv3x3(channels[1] * 2, channels[2], stride=2)
        self.downsample3 = conv3x3(channels[2] * 2, channels[3], stride=2)

        self.ce = CE(norm_layer=Norm2d)

        self.de2 = conv_bn_relu(64, 128, kernel_size=1)
        self.de3 = conv_bn_relu(64, 256, kernel_size=1)
        self.de4 = conv_bn_relu(64, 512, kernel_size=1)

        in_channel = 64
        self.dense_3 = DenseBlock(in_channel * 4, 64, 64, 3, drop_out=0.2)
        self.dense_6 = DenseBlock(in_channel * 5, 64, 64, 6, drop_out=0.2)
        self.dense_9 = DenseBlock(in_channel * 7, 64, 64, 9, drop_out=0.2)
        self.dense_12 = DenseBlock(in_channel * 10, 64, 64, 12, drop_out=0.2)


    def forward(self, features):
        features_t0, features_t1 = features[:4], features[4:]

        #FsGwithCE
        gatee = self.ce(features_t0, features_t1)
        att1 = gatee[0]
        att2 = gatee[1]
        att3 = gatee[2]
        att4 = gatee[3]

        #DFM
        out = torch.cat([att4, att3, att2, att1], dim=1)
        btt4 = self.dense_3(out)
        out = torch.cat([out, btt4], dim=1)
        btt3 = self.dense_6(out)
        out = torch.cat([out, btt4, btt3], dim=1)
        btt2 = self.dense_9(out)
        out = torch.cat([out, btt4, btt3, btt2], dim=1)
        btt1 = self.dense_12(out)

        features_map = torch.cat([out, btt4, btt3, btt2, btt1], dim=1)

        return features_map

class CE(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(CE, self).__init__()

        self.d_in0 = conv_bn_relu(64, 64, 1, norm_layer=norm_layer)
        self.d_in1 = conv_bn_relu(128, 64, 1, norm_layer=norm_layer)
        self.d_in2 = conv_bn_relu(256, 64, 1, norm_layer=norm_layer)
        self.d_in3 = conv_bn_relu(512, 64, 1, norm_layer=norm_layer)
        self.d_in00 = conv_bn_relu(64, 64, 1, norm_layer=norm_layer)
        self.d_in11 = conv_bn_relu(128, 64, 1, norm_layer=norm_layer)
        self.d_in22 = conv_bn_relu(256, 64, 1, norm_layer=norm_layer)
        self.d_in33 = conv_bn_relu(512, 64, 1, norm_layer=norm_layer)

        self.gate0 = conv_sigmoid(64, 64)
        self.gate1 = conv_sigmoid(128, 64)
        self.gate2 = conv_sigmoid(256, 64)
        self.gate3 = conv_sigmoid(512, 64)
        self.gate00 = conv_sigmoid(64, 64)
        self.gate11 = conv_sigmoid(128, 64)
        self.gate22 = conv_sigmoid(256, 64)
        self.gate33 = conv_sigmoid(512, 64)

        self.cls = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True))

        mu0 = torch.Tensor(1, 64, 64)
        mu0.normal_(0, math.sqrt(2. / 64))
        mu0 = self._l2norm(mu0, dim=1)
        self.register_buffer('mu0', mu0)

        mu1 = torch.Tensor(1, 64, 32)
        mu1.normal_(0, math.sqrt(2. / 32))
        mu1 = self._l2norm(mu1, dim=1)
        self.register_buffer('mu1', mu1)

        mu2 = torch.Tensor(1, 64, 16)
        mu2.normal_(0, math.sqrt(2. / 16))
        mu2 = self._l2norm(mu2, dim=1)
        self.register_buffer('mu2', mu2)

        mu3 = torch.Tensor(1, 64, 8)
        mu3.normal_(0, math.sqrt(2. / 8))
        mu3 = self._l2norm(mu3, dim=1)
        self.register_buffer('mu3', mu3)

        self.q0 = torch.nn.Linear(8192, 4096)
        self.q1 = torch.nn.Linear(4096, 2048)
        self.q2 = torch.nn.Linear(2048, 1024)
        self.q3 = torch.nn.Linear(1024, 512)


    def forward(self, x, y):
        m0 = x[0]
        m1 = x[1]
        m2 = x[2]
        m3 = x[3]
        n0 = y[0]
        n1 = y[1]
        n2 = y[2]
        n3 = y[3]

        m0_size = m0.size()[2:]

        #CE

        g_mm0 = self.gate0(m0)
        g_mm1 = self.gate1(m1)
        g_mm2 = self.gate2(m2)
        g_mm3 = self.gate3(m3)
        g_nn0 = self.gate00(n0)
        g_nn1 = self.gate11(n1)
        g_nn2 = self.gate22(n2)
        g_nn3 = self.gate33(n3)


        b_m0, c_m0, h_m0, w_m0 = g_mm0.size()
        b_m1, c_m1, h_m1, w_m1 = g_mm1.size()
        b_m2, c_m2, h_m2, w_m2 = g_mm2.size()
        b_m3, c_m3, h_m3, w_m3 = g_mm3.size()

        g_mm0 = g_mm0.view(b_m0, c_m0, h_m0 * w_m0)
        g_mm1 = g_mm1.view(b_m1, c_m1, h_m1 * w_m1)
        g_mm2 = g_mm2.view(b_m2, c_m2, h_m2 * w_m2)
        g_mm3 = g_mm3.view(b_m3, c_m3, h_m3 * w_m3)
        g_nn0 = g_nn0.view(b_m0, c_m0, h_m0 * w_m0)
        g_nn1 = g_nn1.view(b_m1, c_m1, h_m1 * w_m1)
        g_nn2 = g_nn2.view(b_m2, c_m2, h_m2 * w_m2)
        g_nn3 = g_nn3.view(b_m3, c_m3, h_m3 * w_m3)

        mu0 = self.mu0.repeat(b_m0, 1, 1)
        mu1 = self.mu1.repeat(b_m1, 1, 1)
        mu2 = self.mu2.repeat(b_m2, 1, 1)
        mu3 = self.mu3.repeat(b_m3, 1, 1)

        g_mm0_t = g_mm0.permute(0, 2, 1)
        g_nn0_t = g_nn0.permute(0, 2, 1)
        g_mm1_t = g_mm1.permute(0, 2, 1)
        g_nn1_t = g_nn1.permute(0, 2, 1)
        g_mm2_t = g_mm2.permute(0, 2, 1)
        g_nn2_t = g_nn2.permute(0, 2, 1)
        g_mm3_t = g_mm3.permute(0, 2, 1)
        g_nn3_t = g_nn3.permute(0, 2, 1)

        zm0 = torch.bmm(g_mm0_t, mu0)
        zn0 = torch.bmm(g_nn0_t, mu0)
        zm1 = torch.bmm(g_mm1_t, mu1)
        zn1 = torch.bmm(g_nn1_t, mu1)
        zm2 = torch.bmm(g_mm2_t, mu2)
        zn2 = torch.bmm(g_nn2_t, mu2)
        zm3 = torch.bmm(g_mm3_t, mu3)
        zn3 = torch.bmm(g_nn3_t, mu3)

        zm0 = F.softmax(zm0, dim=2)
        zn0 = F.softmax(zn0, dim=2)
        zm1 = F.softmax(zm1, dim=2)
        zn1 = F.softmax(zn1, dim=2)
        zm2 = F.softmax(zm2, dim=2)
        zn2 = F.softmax(zn2, dim=2)
        zm3 = F.softmax(zm3, dim=2)
        zn3 = F.softmax(zn3, dim=2)

        zm0_ = zm0 / (1e-6 + zm0.sum(dim=1, keepdim=True))
        zn0_ = zn0 / (1e-6 + zn0.sum(dim=1, keepdim=True))
        zm1_ = zm1 / (1e-6 + zm1.sum(dim=1, keepdim=True))
        zn1_ = zn1 / (1e-6 + zn1.sum(dim=1, keepdim=True))
        zm2_ = zm2 / (1e-6 + zm2.sum(dim=1, keepdim=True))
        zn2_ = zn2 / (1e-6 + zn2.sum(dim=1, keepdim=True))
        zm3_ = zm3 / (1e-6 + zm3.sum(dim=1, keepdim=True))
        zn3_ = zn3 / (1e-6 + zn3.sum(dim=1, keepdim=True))


        a0 = torch.bmm(g_mm0, zm0_)
        b_a0, c_a0, k_a0 = a0.size()
        a0 = a0.view(b_a0, c_a0 * k_a0)
        b0 = torch.bmm(g_nn0, zn0_)
        b0 = b0.view(b_a0, c_a0 * k_a0)
        mu0 = torch.cat([a0, b0], dim=1)
        mu0 = self.q0(mu0)
        mu0 = mu0.view(b_a0, c_a0, k_a0)
        mu0 = self._l2norm(mu0, dim=1)

        a1 = torch.bmm(g_mm1, zm1_)
        b_a1, c_a1, k_a1 = a1.size()
        a1 = a1.view(b_a1, c_a1 * k_a1)
        b1 = torch.bmm(g_nn1, zn1_)
        b1 = b1.view(b_a1, c_a1 * k_a1)
        mu1 = torch.cat([a1, b1], dim=1)
        mu1 = self.q1(mu1)
        mu1 = mu1.view(b_a1, c_a1, k_a1)
        mu1 = self._l2norm(mu1, dim=1)

        a2 = torch.bmm(g_mm2, zm2_)
        b_a2, c_a2, k_a2 = a2.size()
        a2 = a2.view(b_a2, c_a2 * k_a2)
        b2 = torch.bmm(g_nn2, zn2_)
        b2 = b2.view(b_a2, c_a2 * k_a2)
        mu2 = torch.cat([a2, b2], dim=1)
        mu2 = self.q2(mu2)
        mu2 = mu2.view(b_a2, c_a2, k_a2)
        mu2 = self._l2norm(mu2, dim=1)

        a3 = torch.bmm(g_mm3, zm3_)
        b_a3, c_a3, k_a3 = a3.size()
        a3 = a3.view(b_a3, c_a3 * k_a3)
        b3 = torch.bmm(g_nn3, zn3_)
        b3 = b3.view(b_a3, c_a3 * k_a3)
        mu3 = torch.cat([a3, b3], dim=1)
        mu3 = self.q3(mu3)
        mu3 = mu3.view(b_a3, c_a3, k_a3)
        mu3 = self._l2norm(mu3, dim=1)


        zm0_t = zm0.permute(0, 2, 1)
        zn0_t = zn0.permute(0, 2, 1)
        zm1_t = zm1.permute(0, 2, 1)
        zn1_t = zn1.permute(0, 2, 1)
        zm2_t = zm2.permute(0, 2, 1)
        zn2_t = zn2.permute(0, 2, 1)
        zm3_t = zm3.permute(0, 2, 1)
        zn3_t = zn3.permute(0, 2, 1)

        g_m0 = mu0.matmul(zm0_t)
        g_n0 = mu0.matmul(zn0_t)
        g_m1 = mu1.matmul(zm1_t)
        g_n1 = mu1.matmul(zn1_t)
        g_m2 = mu2.matmul(zm2_t)
        g_n2 = mu2.matmul(zn2_t)
        g_m3 = mu3.matmul(zm3_t)
        g_n3 = mu3.matmul(zn3_t)

        g_m0 = g_m0.view(b_m0, c_m0, h_m0, w_m0)
        g_n0 = g_n0.view(b_m0, c_m0, h_m0, w_m0)
        g_m1 = g_m1.view(b_m1, c_m1, h_m1, w_m1)
        g_n1 = g_n1.view(b_m1, c_m1, h_m1, w_m1)
        g_m2 = g_m2.view(b_m2, c_m2, h_m2, w_m2)
        g_n2 = g_n2.view(b_m2, c_m2, h_m2, w_m2)
        g_m3 = g_m3.view(b_m3, c_m3, h_m3, w_m3)
        g_n3 = g_n3.view(b_m3, c_m3, h_m3, w_m3)

        g_m0 = F.relu(g_m0, inplace=True)
        g_n0 = F.relu(g_n0, inplace=True)
        g_m1 = F.relu(g_m1, inplace=True)
        g_n1 = F.relu(g_n1, inplace=True)
        g_m2 = F.relu(g_m2, inplace=True)
        g_n2 = F.relu(g_n2, inplace=True)
        g_m3 = F.relu(g_m3, inplace=True)
        g_n3 = F.relu(g_n3, inplace=True)


        m0 = self.d_in0(m0)
        m1 = self.d_in1(m1)
        m2 = self.d_in2(m2)
        m3 = self.d_in3(m3)
        n0 = self.d_in00(n0)
        n1 = self.d_in11(n1)
        n2 = self.d_in22(n2)
        n3 = self.d_in33(n3)

        # FsG
        n00 = (1 + g_n0) * (g_n0 * n0 + Up(g_n1 * n1, size=m0_size) + Up(g_n2 * n2, size=m0_size) + Up(g_n3 * n3,size=m0_size)) + \
              (1 - g_n0) * (g_m0 * m0 + Up(g_m1 * m1, size=m0_size) + Up(g_m2 * m2, size=m0_size) + Up(g_m3 * m3, size=m0_size))

        n11 = Up(1 + g_n1, size=m0_size) * (g_n0 * n0 + Up(g_n1 * n1, size=m0_size) + Up(g_n2 * n2, size=m0_size) + Up(g_n3 * n3, size=m0_size)) + \
              Up(1 - g_n1, size=m0_size) * (g_m0 * m0 + Up(g_m1 * m1, size=m0_size) + Up(g_m2 * m2, size=m0_size) + Up(g_m3 * m3, size=m0_size))


        n22 = Up(1 + g_n2, size=m0_size) * (g_n0 * n0 + Up(g_n1 * n1, size=m0_size) + Up(g_n2 * n2, size=m0_size) + Up(g_n3 * n3, size=m0_size)) + \
              Up(1 - g_n2, size=m0_size) * (g_m0 * m0 + Up(g_m1 * m1, size=m0_size) + Up(g_m2 * m2, size=m0_size) + Up(g_m3 * m3, size=m0_size))


        n33 = Up(1 + g_n3, size=m0_size) * (g_n0 * n0 + Up(g_n1 * n1, size=m0_size) + Up(g_n2 * n2, size=m0_size) + Up(g_n3 * n3, size=m0_size)) + \
              Up(1 - g_n3, size=m0_size) * (g_m0 * m0 + Up(g_m1 * m1, size=m0_size) + Up(g_m2 * m2, size=m0_size) + Up(g_m3 * m3, size=m0_size))


        gatee = [n00, n11, n22, n33]

        return gatee

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))