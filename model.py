###########################################################################
# Created by: Tramac
# Date: 2019-03-25
# Copyright (c) 2017
###########################################################################

"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MACNet', 'get_fast_scnn']


class MACNet(nn.Module):
    def __init__(self, num_classes, aux=False, **kwargs):
        super(MACNet, self).__init__()
        self.aux = aux
        #模块1
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)

        dilation_block_1 = [2]
        self.CFP_Block_1 = nn.Sequential()
        for i in range(0, 1):
            self.CFP_Block_1.add_module("CFP_Module_1_" + str(i), MAFM(64, d=dilation_block_1[i]))

        self.bn_prelu_1 = BNPReLU(64)
        self.bn_prelu_2 = BNPReLU(128)

        #模块2
        self.global_feature_extractor = GlobalFeatureExtractor(65, [64, 96, 128], 128, 6, [3, 3, 3])

        dilation_block_2 = [4]
        self.CFP_Block_2 = nn.Sequential()
        for i in range(0, 1):
            self.CFP_Block_2.add_module("CFP_Module_2_" + str(i), MAFM(128, d=dilation_block_2[i]))

        #模块3
        self.conv = nn.Sequential(
            nn.Conv2d(65, 129, 1),
            nn.BatchNorm2d(129)
        )
        self.feature_fusion = FeatureFusionModule(64, 129, 129)
        # 模块4
        self.classifier = Classifer(354, num_classes)
        self.conv_1 = BasicConv2d(354, 128, kernel_size=3, padding=1)
        # self.conv_2 = BasicConv2d(256, 128, kernel_size=3, padding=1)
        self.semantic_output = nn.Conv2d(128, num_classes, 1)

        if self.aux:
            if self.aux:
                self.aux_head1 = nn.Conv2d(64, num_classes, 1)
                self.aux_head2 = nn.Conv2d(96, num_classes, 1)
                self.aux_head3 = nn.Conv2d(129, num_classes, 1)

        self.conv1 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(128, 1, kernel_size=1, stride=1)

    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x)
        #print(higher_res_features.size())  #[18, 64, 64, 64]

        bd1 = self.conv1(higher_res_features)

        output1 = self.CFP_Block_1(higher_res_features)
        output1 = self.bn_prelu_1(output1)

        output1 = torch.cat([output1, bd1], 1)

        x_20, x_10, x_5 = self.global_feature_extractor(output1)

        res_2 = F.interpolate(x_20, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_10, size=x.size()[2:], mode='bilinear', align_corners=True)


        bd2 = self.conv2(x_5)
        #inception模块
        output2 = self.CFP_Block_2(x_5)
        output2 = self.bn_prelu_2(output2)
        output2 = torch.cat([output2, bd2], 1)

        res_4 = F.interpolate(output2, size=x.size()[2:], mode='bilinear', align_corners=True)

        res = [output1, res_2, res_3, res_4]
        out = torch.cat(res, dim=1)

        out = self.conv_1(out)
        # out = self.conv_2(out)
        out = self.semantic_output(out)
        out = F.softmax(out, dim=1)
        #print(x.shape)
        if self.aux:
            res_2 = self.aux_head1(res_2)
            res_2 = F.softmax(res_2, dim=1)

            res_3 = self.aux_head2(res_3)
            res_3 = F.softmax(res_3, dim=1)

            res_4 = self.aux_head3(res_4)
            res_4 = F.softmax(res_4, dim=1)

        if self.aux:
            return [out, res_2, res_3, res_4]
        else:
            return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(5, dw_channels1, 3, 1, padding=1)
        self.conv1 = _ConvBNReLU(dw_channels1, dw_channels2, 3, 1, padding=1)
        self.conv2 = _ConvBNReLU(dw_channels2, out_channels, 3, 1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output



class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 2)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_20 = self.bottleneck1(x)
        #print(x.shape)
        x_10 = self.bottleneck2(x_20)
        #print(x.size())
        x_5 = self.bottleneck3(x_10)
        #print('111')
        #print(x.shape)
        #x = self.ppm(x)
        #print(x.shape)
        return x_20, x_10, x_5


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        #lower_res_feature = self.dwconv(lower_res_feature)
        #lower_res_feature = self.conv_lower_res(lower_res_feature)

        #higher_res_feature = self.conv_higher_res(higher_res_feature)
        #print(higher_res_feature.shape)
        #print(lower_res_feature.shape)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, 128, stride)
        self.dsconv2 = _DSConv(128, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(130, num_classes, 1)
        )

        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.dsconv1(x)

        bd3 = self.conv3(x)
        #print(bd3.shape)
        x = self.dsconv2(x)

        x = torch.cat([x, bd3], 1)
        #print(x.shape)
        x = self.conv(x)
        return x


def get_fast_scnn(dataset='citys', pretrained=False, root='./weights', map_cpu=False, **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    #from data_loader import datasets
    model = MACNet(2, **kwargs)
    if pretrained:
        if(map_cpu):
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
        else:
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
    return model


if __name__ == '__main__':
    img = torch.randn(2, 3, 256, 512)
    model = MACNet('citys')
    outputs = model(img)
