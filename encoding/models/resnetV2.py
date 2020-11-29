import torch
import torch.nn as nn
import math
import os
from collections import OrderedDict

BN = None

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        bypass_bn_weight_list.append(self.bn3.weight)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_stem=False,
                 avg_down=False, bypass_last_bn=False,
                 bn=None):

        super(ResNet, self).__init__()


        global BN, bypass_bn_weight_list

        BN = bn
        bypass_bn_weight_list = []

        self.inplanes = 64
        self.deep_stem = deep_stem
        self.avg_down = avg_down

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                        BN(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        BN(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()

        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    BN(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BN(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


#def resnet50(**kwargs):
def resnet50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
         #model.load_state_dict(torch.load(
        #    get_model_file('resnet101', root=root)), strict=False)
        if not os.path.isfile(root):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(root))
        checkpoint = torch.load(root)
        #old_state_dict = checkpoint['state_dict']
        #old_state_dict = checkpoint['model']
        if isinstance(checkpoint, OrderedDict):
             old_state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
             old_state_dict = checkpoint['model']
        else:
           raise RuntimeError('No state_dict found in file {}'.format(root))
        #old_model = checkpoint['model']
        #old_state_dict = old_model['state_dict']
        new_state_dict = dict()
        for k, v in old_state_dict.items():
            print("Copy weights at {}".format(k))
            if k.startswith('module.'):
                new_state_dict[k[len('module.'):]] = old_state_dict[k]
            else:
                new_state_dict[k] = old_state_dict[k]

        model.load_state_dict(new_state_dict, strict=False)

    return model

def resnet101(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(torch.load(
        #    get_model_file('resnet101', root=root)), strict=False)
        if not os.path.isfile(root):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(root))
        checkpoint = torch.load(root)
        #old_state_dict = checkpoint['state_dict']
        #old_state_dict = checkpoint['model']
        if isinstance(checkpoint, OrderedDict):
             old_state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
             old_state_dict = checkpoint['model']
        else:
           raise RuntimeError('No state_dict found in file {}'.format(root))
        #old_model = checkpoint['model']
        #old_state_dict = old_model['state_dict']
        new_state_dict = dict()
        for k, v in old_state_dict.items():
            print("Copy weights at {}".format(k))
            if k.startswith('module.'):
                new_state_dict[k[len('module.'):]] = old_state_dict[k]
            else:
                new_state_dict[k] = old_state_dict[k]

        model.load_state_dict(new_state_dict, strict=False)

    return model

#def resnet101(**kwargs):
#    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#    return model


#def resnet152(**kwargs):
def resnet152(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    if pretrained:
        #model.load_state_dict(torch.load(
        #    get_model_file('resnet101', root=root)), strict=False)
        if not os.path.isfile(root):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(root))
        checkpoint = torch.load(root)
        #old_state_dict = checkpoint['state_dict']
        if isinstance(checkpoint, OrderedDict):
             old_state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
             old_state_dict = checkpoint['model']
        else:
           raise RuntimeError('No state_dict found in file {}'.format(root))
        #old_model = checkpoint['model']
        #old_state_dict = old_model['state_dict']
        new_state_dict = dict()
        for k, v in old_state_dict.items():
            print("Copy weights at {}".format(k))
            if k.startswith('module.'):
                new_state_dict[k[len('module.'):]] = old_state_dict[k]
            else:
                new_state_dict[k] = old_state_dict[k]

        model.load_state_dict(new_state_dict, strict=False)


    return model
