###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter

#from . import resnet
from . import resnetV2
from ..utils import batch_pix_accuracy, batch_intersection_union

#up_kwargs = {'mode': 'bilinear', 'align_corners': True}
up_kwargs = {'mode': 'bilinear', 'align_corners': False}

__all__ = ['BaseNet', 'MultiEvalModule', 'MultiEvalModule2']

class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, aux, se_loss=False, dilated=True, norm_layer=None, 
                 multi_grid=False,
                 base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='~/.encoding/models'):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        # copying modules from pretrained models
        self.backbone = backbone
        if backbone == 'resnet50':
            self.pretrained = resnetV2.resnet50(pretrained=True,
                                              deep_stem = True, avg_down=True,
                                              bn = norm_layer, root=root)
        elif backbone == 'resnet101':
            self.pretrained = resnetV2.resnet101(pretrained=True,
                                              deep_stem = True, avg_down=True,
                                              bn = norm_layer, root=root)
        elif backbone == 'resnet152':
            self.pretrained = resnetV2.resnet152(pretrained=True,
                                              deep_stem = True, avg_down=True,
                                              bn = norm_layer, root=root)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        if self.backbone.startswith('wideresnet'):
            x = self.pretrained.mod1(x)
            x = self.pretrained.pool2(x)
            x = self.pretrained.mod2(x)
            x = self.pretrained.pool3(x)
            x = self.pretrained.mod3(x)
            x = self.pretrained.mod4(x)
            x = self.pretrained.mod5(x)
            c3 = x.clone()
            x = self.pretrained.mod6(x)
            x = self.pretrained.mod7(x)
            x = self.pretrained.bn_out(x)
            return None, None, c3, x
        else:
            #x1 = self.pretrained.conv1(x)
            #x2 = self.pretrained.bn1(x1)
            #x3 = self.pretrained.relu(x2)
            #x4 = self.pretrained.maxpool(x3)
            #c1 = self.pretrained.layer1(x4)
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)
        #print(f"x shape: {x.shape}, mean: {x.mean()}")
        #print(f"x1 shape: {x1.shape}, mean: {x1.mean()}")
        #print(f"x2 shape: {x2.shape}, mean: {x2.mean()}")
        #print(f"x3 shape: {x3.shape}, mean: {x3.mean()}")
        #print(f"x4 shape: {x4.shape}, mean: {x4.mean()}")
        #print(f"c1 shape: {c1.shape}, mean: {c1.mean()}")
        #print(f"c2 shape: {c2.shape}, mean: {c2.mean()}")
        #print(f"c3 shape: {c3.shape}, mean: {c3.mean()}")
        #print(f"c4 shape: {c4.shape}, mean: {c4.mean()}")
        #x_nan_mask = torch.isnan(x)
        #if x_nan_mask.any():
        #    print("NaN-Found-In-x-Layer.")
        #x1_nan_mask = torch.isnan(x1)
        #if x1_nan_mask.any():
        #    print("NaN-Found-In-x1-Layer.")
        #x2_nan_mask = torch.isnan(x2)
        #if x2_nan_mask.any():
        #    print("NaN-Found-In-x2-Layer.")
        #x3_nan_mask = torch.isnan(x3)
        #if x3_nan_mask.any():
        #    print("NaN-Found-In-x3-Layer.")
        #x4_nan_mask = torch.isnan(x4)
        #if x4_nan_mask.any():
        #    print("NaN-Found-In-x4-Layer.")
        return c1, c2, c3, c4

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union


class MultiEvalModule(DataParallel):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, device_ids=None, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        super(MultiEvalModule, self).__init__(module, device_ids)
        self.nclass = nclass
        self.base_size = module.base_size
        self.crop_size = module.crop_size
        self.scales = scales
        self.flip = flip
        print('MultiEvalModule: base_size {}, crop_size {}'. \
            format(self.base_size, self.crop_size))

    def parallel_forward(self, inputs, **kwargs):
        """Multi-GPU Mult-size Evaluation

        Args:
            inputs: list of Tensors
        """
        inputs = [(input.unsqueeze(0).cuda(device),)
                  for input, device in zip(inputs, self.device_ids)]
        replicas = self.replicate(self, self.device_ids[:len(inputs)])
        kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        #for out in outputs:
        #    print('out.size()', out.size())
        return outputs

    def forward(self, image):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        batch, _, h, w = image.size()
        assert(batch == 1)
        stride_rate = 2.0/3.0
        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch,self.nclass,h,w).zero_().cuda()

        for scale in self.scales:
            if h > w:
                long_size_base = h
            else:
                long_size_base = w
            #long_size = int(math.ceil(self.base_size * scale))
            #long_size = int(math.ceil(long_size_base * scale))
            #long_size = math.ceil(long_size_base * scale)
            long_size = (long_size_base * scale)
            #long_size = math.floor(long_size / 32.0 + 0.5) * 32.0
            if h > w:
                height = long_size
                #width = int(1.0 * w * long_size / h + 0.5)
                width = (1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                #height = int(1.0 * h * long_size / w + 0.5)
                height = (1.0 * h * long_size / w + 0.5)
                short_size = height
            """
            short_size = int(math.ceil(self.base_size * scale))
            if h > w:
                width = short_size
                height = int(1.0 * h * short_size / w)
                long_size = height
            else:
                height = short_size
                width = int(1.0 * w * short_size / h)
                long_size = width
            """
            #height = int(math.ceil(height / 8) * 8)
            #width = int(math.ceil(width / 8) * 8)
            #height = int(math.floor(height / 8 + 0.5) * 8)
            #width = int(math.floor(width / 8 + 0.5) * 8)
            height = int(math.floor(height / 32.0 + 0.5) * 32)
            width = int(math.floor(width / 32.0 + 0.5) * 32)
            #height = int(height)
            #width = int(width)
            numPixel = int(height * width)
            #if (numPixel > 2200000):
            #if (numPixel > 2000000):
            #if (numPixel > 2500000):
            #if (numPixel > 3000000):
            #if (numPixel > 5000000):
            if (numPixel > 3000000):
                break
            # resize image to current size
            cur_img = resize_image(image, height, width, **self.module._up_kwargs)
            outputs = module_inference(self.module, cur_img, self.flip)
            #out_img = self.module.evaluate(cur_img)
            #cur_img_flip = flip_image(cur_img)
            #out_img_flip = self.module.evaluate(cur_img_flip)
            #out_img_flip = flip_image(out_img_flip)
            #out_img = out_img.exp()
            #out_img_flip = out_img_flip.exp()
            #outputs = out_img + out_img_flip
            #outputs = outputs
            score = resize_image(outputs, h, w, **self.module._up_kwargs)
            #score = F.softmax(score, dim=1)
            #score = F.softmax(score, dim=1)
            scores += score 

        return scores

class MultiEvalModule2(DataParallel):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, device_ids=None, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        super(MultiEvalModule2, self).__init__(module, device_ids)
        self.nclass = nclass
        self.base_size = module.base_size
        self.crop_size = module.crop_size
        self.scales = scales
        self.flip = flip
        print('MultiEvalModule2: base_size {}, crop_size {}'. \
            format(self.base_size, self.crop_size))

    def parallel_forward(self, inputs, **kwargs):
        """Multi-GPU Mult-size Evaluation

        Args:
            inputs: list of Tensors
        """
        inputs = [(input.unsqueeze(0).cuda(device),)
                  for input, device in zip(inputs, self.device_ids)]
        replicas = self.replicate(self, self.device_ids[:len(inputs)])
        kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        #for out in outputs:
        #    print('out.size()', out.size())
        return outputs

    def forward(self, image):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        batch, _, h, w = image.size()
        assert(batch == 1)
        stride_rate = 2.0/3.0
        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch,self.nclass,h,w).zero_().cuda()

        for scale in self.scales:
            if h > w:
                long_size_base = h
            else:
                long_size_base = w
            long_size = int(math.ceil(self.base_size * scale))
            #long_size = int(math.ceil(long_size_base * scale))
            #long_size = math.ceil(long_size_base * scale)
            #long_size = (long_size_base * scale)
            #long_size = math.floor(long_size / 32.0 + 0.5) * 32.0
            if h > w:
                height = long_size
                #width = int(1.0 * w * long_size / h + 0.5)
                width = (1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                #height = int(1.0 * h * long_size / w + 0.5)
                height = (1.0 * h * long_size / w + 0.5)
                short_size = height
            """
            short_size = int(math.ceil(self.base_size * scale))
            if h > w:
                width = short_size
                height = int(1.0 * h * short_size / w)
                long_size = height
            else:
                height = short_size
                width = int(1.0 * w * short_size / h)
                long_size = width
            """
            #height = int(math.ceil(height / 8) * 8)
            #width = int(math.ceil(width / 8) * 8)
            #height = int(math.floor(height / 8 + 4.0) * 8)
            #width = int(math.floor(width / 8 + 4.0) * 8)
            height = int(math.floor(height / 32.0 + 0.5) * 32)
            width = int(math.floor(width / 32.0 + 0.5) * 32)
            #height = int(height)
            #width = int(width)
            numPixel = int(height * width)
            #if (numPixel > 2200000):
            #if (numPixel > 2000000):
            #if (numPixel > 2500000):
            #if (numPixel > 5000000):
            if (numPixel > 8000000):
                break
            # resize image to current size
            cur_img = resize_image(image, height, width, **self.module._up_kwargs)
            outputs = module_inference(self.module, cur_img, self.flip)
            #out_img = self.module.evaluate(cur_img)
            #cur_img_flip = flip_image(cur_img)
            #out_img_flip = self.module.evaluate(cur_img_flip)
            #out_img_flip = flip_image(out_img_flip)
            #out_img = out_img.exp()
            #out_img_flip = out_img_flip.exp()
            #outputs = out_img + out_img_flip
            #outputs = outputs
            score = resize_image(outputs, h, w, **self.module._up_kwargs)
            #score = F.softmax(score, dim=1)
            #score = F.softmax(score, dim=1)
            scores += score 

        return scores


def module_inference(module, image, flip=True):
    output = module.evaluate(image)
    output = output.exp()
    #output = F.softmax(output, dim=1)
    if flip:
        fimg = flip_image(image)
        foutput = module.evaluate(fimg)
        foutput = foutput.exp()
        output += flip_image(foutput)
    return output

def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)

def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)
