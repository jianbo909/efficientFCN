###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import math
from .base import BaseNet
from .fcn import FCNHead
#from ..nn import SyncBatchNorm, Encoding, Mean, GlobalAvgPool2d
from ..nn import Mean, GlobalAvgPool2d

__all__ = ['efficientFCN', 'HGDModule', 'get_efficientfcn', 'get_efficientfcn_resnet50_pcontext',
           'get_efficientfcn_resnet101_pcontext', 'get_efficientfcn_resnet50_citys',
           'get_efficientfcn_resnet101_citys', 'get_efficientfcn_resnet50_ade',
           'get_efficientfcn_resnet101_ade']

class efficientFCN(BaseNet):
    def __init__(self, nclass, backbone, num_center=256, aux=True, norm_layer=None, **kwargs):
        super(efficientFCN, self).__init__(nclass, backbone, aux, dilated=False,
                                     norm_layer=norm_layer, **kwargs)
        self.head = HGDecoder(2048, self.nclass, num_center=num_center,
                            norm_layer=norm_layer,
                            up_kwargs=self._up_kwargs)
        #if aux:
        #    self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)

        x = list(self.head(*features))
        x[0] = F.interpolate(x[0], imsize, **self._up_kwargs)
        #x[2] = F.interpolate(x[2], imsize, **self._up_kwargs)
        #if self.aux:
        #    #auxout = self.auxlayer(features[2])
        #    #auxout = F.interpolate(auxout, imsize, **self._up_kwargs)
        #    x[1] = F.interpolate(x[1], imsize, **self._up_kwargs)
        #    #x.append(x[1])
        #    #x[2] = F.interpolate(x[2], imsize, **self._up_kwargs)
        #    #x.append(x[2])
        return tuple(x)


class HGDModule(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None):
        super(HGDModule, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat= nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_center= nn.Sequential(
            nn.Conv2d(in_channels*3, center_channels, 1, bias=False),
            #norm_layer(out_channels),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels))
        self.norm_center= nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0 = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels , 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.conv_affinity1 = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            nn.ReLU(inplace=True))
        self.conv_up = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, norm_layer):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

    def forward(self, x, guide1, guide2):
        n, c, h, w = x.size()
        n1, c1, h1, w1 = guide1.size()
        n2, c2, h2, w2 = guide2.size()
        x_up0 = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=True)
        x_up1 = F.interpolate(guide1, size=(h2, w2), mode='bilinear', align_corners=True)
        guide1_down = F.interpolate(guide1, size=(h, w), mode='bilinear', align_corners=True)
        guide2_down = F.interpolate(guide2, size=(h, w), mode='bilinear', align_corners=True)
        x_cat = torch.cat([guide2_down, guide1_down, x], 1)
        guide_cat = torch.cat([guide2, x_up1,  x_up0], 1)
        f_cat = self.conv_cat(x_cat)
        f_center = self.conv_center(x_cat)
        f_cat = f_cat.view(n, self.out_channels, h*w)
        #f_x = x_cat.view(n, 2*c, h*w)
        f_center_norm = f_center.view(n, self.center_channels, h*w)
        f_center_norm = self.norm_center(f_center_norm)
        #n x * in_channels x center_channels
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))
        

        ########################################
        f_cat = f_cat.view(n, self.out_channels, h, w)
        f_cat_avg = self.avgpool0(f_cat)
        value_avg = f_cat_avg.repeat(1, 1, h2, w2)

        ###################################
        #f_affinity = self.conv_affinity(guide_cat)
        guide_cat_conv = self.conv_affinity0(guide_cat)
        guide_cat_value_avg = guide_cat_conv + value_avg
        f_affinity = self.conv_affinity1(guide_cat_value_avg)
        n_aff, c_ff, h_aff, w_aff = f_affinity.size()
        f_affinity = f_affinity.view(n_aff, c_ff, h_aff * w_aff)
        norm_aff = ((self.center_channels) ** -.5)
        #x_up = norm_aff * x_center.bmm(f_affinity.transpose(1, 2))
        x_up = norm_aff * x_center.bmm(f_affinity)
        x_up = x_up.view(n, self.out_channels, h_aff, w_aff)
        x_up_cat = torch.cat([x_up, guide_cat_conv], 1)
        x_up_conv = self.conv_up(x_up_cat)
        outputs = [x_up_conv]
        return tuple(outputs)


class HGDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_center,
                 norm_layer=None, up_kwargs=None):
        super(HGDecoder, self).__init__()
        self.up_kwargs = up_kwargs
        self.conv50 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1, padding=0, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True))
        self.conv40 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, padding=0, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True))
        #self.conv30 = nn.Sequential(
        #    nn.Conv2d(512, 512, 1, padding=0, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        #self.conv52 = nn.Sequential(
        #    nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        #self.conv53 = nn.Sequential(
        #    nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        #self.conv51 = nn.Sequential(
        #    nn.Conv2d((512+out_channels), 512, 1, bias=False),
        #    norm_layer(512),
        #    nn.ReLU(inplace=True))
        #self.num_center = 128
        #self.num_center = 256
        #self.num_center = int(out_channels * 4)
        #self.num_center = out_channels
        #self.num_center = 600
        self.num_center = num_center
        self.hgdmodule0 = HGDModule(512, self.num_center, 1024, norm_layer=norm_layer)
        self.conv_pred3 = nn.Sequential(nn.Dropout2d(0.1, False),
            nn.Conv2d(1024, out_channels, 1, padding=0))

        #for m in self.modules():
        #    #print(f"initialize {m} layer.")
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, norm_layer):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

               

    def forward(self, *inputs):
        feat50 = self.conv50(inputs[-1])
        feat40 = self.conv40(inputs[-2])
        #feat30 = self.conv30(inputs[-3])
        outs0 = list(self.hgdmodule0(feat50, feat40, inputs[-3]))
        outs_pred3 = self.conv_pred3(outs0[0])
        outs = [outs_pred3]

        return tuple(outs)


def get_efficientfcn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
               root='~/.encoding/models', **kwargs):
    r"""EncNet model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    backbone : str, default resnet50
        The backbone network. (resnet50, 101, 152)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    #kwargs['lateral'] = True if dataset.lower().startswith('p') else False
    # infer number of classes
    from ..datasets import datasets, acronyms
    model = efficientFCN(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('efficientFCN_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_efficientfcn_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('pcontext', 'resnet50', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

def get_efficientfcn_resnet101_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet101_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('pcontext', 'resnet101', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

def get_efficientfcn_resnet50_citys(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet50_citys(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('citys', 'resnet50', pretrained, root=root, aux=True,
                      base_size=1024, crop_size=768, **kwargs)

def get_efficientfcn_resnet101_citys(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet101_citys(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('citys', 'resnet101', pretrained, root=root, aux=True,
                      base_size=1024, crop_size=768, **kwargs)


def get_efficientfcn_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('ade20k', 'resnet50', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)

def get_efficientfcn_resnet101_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('ade20k', 'resnet101', pretrained, root=root, aux=True,
                      base_size=640, crop_size=576, **kwargs)

def get_efficientfcn_resnet152_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_efficientfcn_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_efficientfcn('ade20k', 'resnet152', pretrained, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)
