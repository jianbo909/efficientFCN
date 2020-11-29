import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

import torch
from .base import BaseDataset

class VOCSegmentation(BaseDataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]
    NUM_CLASS = 21
    BASE_DIR = 'VOCdevkit/VOC2012'
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(VOCSegmentation, self).__init__(root, split, mode, transform,
                                              target_transform, **kwargs)
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        #_mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _mask_dir = os.path.join(_voc_root, 'SegmentationClassAug')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        #if self.mode == 'train':
        if self.split == 'train':
            #_split_f = os.path.join(_splits_dir, 'trainval.txt')
            _split_f = os.path.join(_splits_dir, 'trainaug.txt')
        #elif self.mode == 'val':
        elif self.split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        #elif self.mode == 'test':
        elif self.split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            #img, target = self._sync_transform( img, target, 255)
            img, target = self._sync_transform( img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform( img, target)
        else:
            assert self.mode == 'testval'
            #mask = self._mask_transform(mask)
            target = self._mask_transform(target)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return img, self._mask_transform(mask)

    #def _sync_transform(self, img, mask, mask_val=0):
    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        w, h = img.size
        if h > w:
            long_size_base = h
        else:
            long_size_base = w
        long_size = random.randint(int(long_size_base*0.5), int(long_size_base*2.0))
        #long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
            #mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=mask_val)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # final transform
        return img, self._mask_transform(mask)


    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0
