###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
#from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.nn import SegmentationLosses
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule, MultiEvalModule2

from tensorboardX import SummaryWriter
from option import Options

def test(args):
    directory = "runs/val_summary/%s/%s/%s/"%(args.dataset, args.model, args.resume)
    if not os.path.exists(directory):
        os.makedirs(directory)
    writer = SummaryWriter(directory)
    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    if args.eval:
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval',
                                           transform=input_transform)
    elif args.test_val:
        testset = get_segmentation_dataset(args.dataset, split='val', mode='test',
                                           transform=input_transform)
    else:
        testset = get_segmentation_dataset(args.dataset, split='test', mode='test',
                                           transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)

    Norm_method = torch.nn.BatchNorm2d
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
        #model.base_size = args.base_size
        #model.crop_size = args.crop_size
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone = args.backbone, aux = args.aux,
                                       multi_grid = args.multi_grid,
                                       num_center = args.num_center,
                                       norm_layer = Norm_method,
                                       root = args.backbone_path,
                                       base_size=args.base_size, crop_size=args.crop_size)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        #model.module.load_state_dict(checkpoint['state_dict'])
        old_state_dict = checkpoint['state_dict']
        new_state_dict = dict()
        for k, v in old_state_dict.items():
            if k.startswith('module.'):
                #new_state_dict[k[len('module.'):]] = old_state_dict[k]
                new_state_dict[k[len('model.module.'):]] = old_state_dict[k]
                #new_state_dict[k] = old_state_dict[k]
            else:
                new_state_dict[k] = old_state_dict[k]
                #new_k = 'module.' + k
                #new_state_dict[new_k] = old_state_dict[k]
 
        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    print(model)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
        [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    if args.dataset == 'ade20k':
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    if not args.ms:
        scales = [1.0]
    
    if args.dataset == 'ade20k':
         evaluator = MultiEvalModule2(model, testset.num_class, scales=scales, flip=args.ms).cuda()
    else:
         evaluator = MultiEvalModule(model, testset.num_class, scales=scales, flip=args.ms).cuda()

    evaluator.eval()
    metric = utils.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    for i, (image, dst) in enumerate(tbar):
        if args.eval:
            with torch.no_grad():
                predicts = evaluator.parallel_forward(image)
                metric.update(dst, predicts)
                pixAcc, mIoU = metric.get()
                tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
                writer.add_scalar('pixAcc', pixAcc, i)
                writer.add_scalar('mIoU', mIoU, i)
        else:
            with torch.no_grad():
                outputs = evaluator.parallel_forward(image)
                predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                            for output in outputs]
            for predict, impath in zip(predicts, dst):
                mask = utils.get_mask_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))
    writer.close()

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    test(args)
