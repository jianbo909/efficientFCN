###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import copy
import numpy as np

import torch
import torch.nn as nn

#if __name__ == '__main__':
#    torch.multiprocessing.set_start_method('spawn')

from tqdm import tqdm
#from multiprocessing import Lock
#tqdm.set_lock(Lock())  # manually set internal lock

from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
#from encoding.nn import SegmentationLosses, SyncBatchNorm, OHEMSegmentationLosses
#from encoding.nn import SegmentationLosses, SyncBatchNorm2d, OHEMSegmentationLosses
from encoding.nn import SegmentationLosses, OHEMSegmentationLosses
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_dataset, test_batchify_fn
from encoding.models import get_segmentation_model

###add tensorboarX
from tensorboardX import SummaryWriter
###support distributed train
import torch.distributed as dist
import subprocess

from option import Options

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

def init_dist_slurm(args, backend='nccl'):
    proc_id = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = '19626'
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    #total_gpus =dist.get_world_size()
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    assert args.batch_size % num_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (args.batch_size, num_gpus)
    #args.batch_size = args.batch_size // num_gpus
    #args.local_rank = proc_id % num_gpus
    #torch.cuda.set_device(rank % num_gpus)
    torch.cuda.set_device(local_rank)
    #os.environ['MASTER_PORT'] = str(args.tcp_port)
    #os.environ['MASTER_PORT'] = '16666'
    dist.init_process_group(backend=backend)
    assert dist.is_initialized()
    #return world_size
    return local_rank

def broadcast_params(model):
    for p in model.state_dict().values():
        dist.broadcast(p, 0)


def average_gradients(model):
    for param in model.parameters():
        if param.requires_grad and not (param.grad is None):
            dist.all_reduce(param.grad.data)

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels):
    outputs = self.model(inputs)
    loss = self.loss(outputs, labels)
    return torch.unsqueeze(loss,0), outputs



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





class Trainer():
    def __init__(self, args):
        self.args = args
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
        trainset = get_dataset(args.dataset, split=args.train_split, mode='train',
                                           **data_kwargs)
        testset = get_dataset(args.dataset, split='val', mode ='val',
                                           **data_kwargs)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        #self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
        #                                   drop_last=True, shuffle=True, **kwargs)
        #collate_fn=test_batchify_fn,
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size // args.world_size,
                                           drop_last=True, shuffle=False, sampler=self.train_sampler, **kwargs)
        #self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
        self.valloader = data.DataLoader(testset, batch_size=args.test_batch_size // args.world_size,
                                         drop_last=False, shuffle=False, sampler=self.val_sampler, **kwargs)
        self.nclass = trainset.num_class
        #Norm_method = nn.SyncBatchNorm 
        #Norm_method = nn.BatchNorm2d(momentum=0.01) 
        Norm_method = nn.BatchNorm2d 
        # model
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone = args.backbone, aux = args.aux,
                                       multi_grid = args.multi_grid,
                                       se_loss = args.se_loss, norm_layer = Norm_method,
                                       lateral = args.lateral,
                                       root = args.backbone_path,
                                       base_size=args.base_size, crop_size=args.crop_size)
        if self.args.rank == 0:
            print(model)
        
        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
        optimizer = torch.optim.SGD(params_list, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer = optimizer

        #self.model = model
        # criterions
        self.criterion = SegmentationLosses(se_loss=args.se_loss,
                                            aux=args.aux,
                                            nclass=self.nclass, 
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight)
        

        device = torch.device('cuda:{}'.format(args.local_rank))

        self.device = device
        # using cuda
        if args.cuda:
            #self.model = DataParallelModel(self.model).cuda()
            #self.model = self.model.cuda()
            sync_bn_model = FullModel(model, self.criterion)
            #self.model.cuda()
            #broadcast_params(self.model)
            #num_gpus = torch.cuda.device_count()
            #local_rank = args.local_rank % num_gpus
            #local_rank = args.local_rank
            #process_group = torch.distributed.new_group([args.local_rank])
            #process_group = torch.distributed.new_group([args.rank])
            #sync_bn_model = torch.nn.utils.convert_sync_batchnorm(self.model, process_group)
            sync_bn_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(sync_bn_model)
            sync_bn_model = sync_bn_model.to(device)
            #self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(sync_bn_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
            #self.criterion = DataParallelCriterion(self.criterion).cuda()
            #self.criterion = self.criterion.cuda()
            dist.barrier()            
        
        # resuming checkpoint
        #if args.resume is not None and self.args.rank == 0:
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            old_state_dict = checkpoint['state_dict']
            new_state_dict = dict()
            for k, v in old_state_dict.items():
                if k.startswith('module.'):
                    #new_state_dict[k[len('module.'):]] = old_state_dict[k]
                    new_state_dict[k] = old_state_dict[k]
                else:
                    new_state_dict[k] = old_state_dict[k]

            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                #self.model.module.load_state_dict(checkpoint['state_dict'])
                #self.model.load_state_dict(checkpoint['state_dict'])
                self.model.load_state_dict(new_state_dict)
            else:
                #self.model.load_state_dict(checkpoint['state_dict'])
                self.model.load_state_dict(new_state_dict)
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda() 

            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))


        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader), local_rank=self.args.rank)
        print('len(trainloader) : %.3f ' % (len(self.trainloader)))


        self.best_pred = 0.0
        #for sumaryWriter
        self.track_loss = 0.0
        self.track_pixAcc = 0.0
        self.track_mIoU = 0.0

    def training(self, epoch):
        #print('Training: 0')
        self.train_sampler.set_epoch(epoch)
        #train_loss = 0.0
        world_size = self.args.world_size 
        losses = AverageMeter()

        self.model.train()
        #print('Training: 1')
        tbar = tqdm(self.trainloader, disable=self.args.rank not in [0])
        #tbar = self.trainloader
        #if args.local_rank == 0:
        #    print('Training: 2')
        for i, (image, target) in enumerate(tbar):
            #if args.local_rank == 0:
            #    print('Training: 3')
            #self.optimizer.zero_grad()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            #if args.local_rank == 0:
            #    print('Training: 4')
            #if torch_ver == "0.3":
            #    image = Variable(image)
            #    target = Variable(target)
            #target = target.cuda(non_blocking=True)
            #image = torch.autograd.Variable(image.cuda(non_blocking=True))
            #target = torch.autograd.Variable(target)
            image = image.to(self.device)
            target = target.to(self.device)
            loss_out, _ = self.model(image, target)

            #loss = self.criterion(outputs, target)
            loss = loss_out.mean()
            reduced_loss = loss.data.clone() 
            reduced_loss = reduced_loss / world_size 
            #reduced_loss = loss
            dist.all_reduce_multigpu([reduced_loss])
            #print('rank = %.3f(%.3f) ---> Loss =  %.3f.' % (self.args.local_rank, self.args.rank, loss))
            #losses.update(reduced_loss.item(), image.size(0))
            losses.update(reduced_loss.item(), 1)

            self.model.zero_grad()
            loss.backward()
            #average_gradients(self.model)
            self.optimizer.step()
            
            dist.barrier()            
 
            #train_loss += loss.item()
            self.track_loss = losses.avg
            if self.args.rank == 0:
                #tbar.set_description('Train loss: %.3f, in_max: %.3f, in_min: %.3f, out_max: %.3f, out_min: %.3f, gt_max: %.3f, gt_min: %.3f, loss0: %.3f, ws: %.3f, ns: %.3f' % (losses.avg, torch.max(image).item(), torch.min(image).item(), torch.max(outputs[0]).item(), torch.min(outputs[0]).item(), torch.max(target).item(), torch.min(target).item(), loss.item(), world_size, image.size(0)))
                tbar.set_description('Train loss: %.3f'% (losses.avg))

        if self.args.no_val and self.args.rank == 0:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                #'state_dict': self.model.module.state_dict(),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            #image = image.cuda(non_blocking=True)
            #target = target.cuda(non_blocking=True)
            image = image.to(self.device)
            target = target.to(self.device)
            _, outputs = model(image, target)
            #outputs = model(image)
            #correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            #inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            correct, labeled = utils.batch_pix_accuracy(outputs, target)
            inter, union = utils.batch_intersection_union(outputs, target, self.nclass)
            return correct, labeled, inter, union

        world_size = self.args.world_size
        is_best = False
        self.model.eval()
        #total_inter = AverageMeter()
        #total_union = AverageMeter()
        total_inter = 0
        total_union = 0
        total_correct = AverageMeter()
        total_label = AverageMeter()
        tbar = tqdm(self.valloader, desc='\r', disable=self.args.rank not in [0])
        for i, (image, target) in enumerate(tbar):
            #target = target.cuda()
            #image_var = torch.autograd.Variable(image.cuda(), volatile=True)
            #target = torch.autograd.Variable(target, volatile=True)

            with torch.no_grad():
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            
            #reduced_correct = correct.clone() / world_size
            #reduced_label = label.clone() / world_size
            #reduced_inter = inter.clone() / world_size
            #reduced_union = union.clone() / world_size
            inter = inter.cuda()
            union = union.cuda()
            #reduced_correct = correct.data.clone() / world_size
            #reduced_label = labeled.data.clone() / world_size
            reduced_correct = correct.data.clone()
            reduced_label = labeled.data.clone()
            reduced_inter = inter.data.clone() 
            reduced_union = union.data.clone() 
            dist.all_reduce_multigpu([reduced_correct])
            dist.all_reduce_multigpu([reduced_label])
            dist.all_reduce_multigpu([reduced_inter])
            dist.all_reduce_multigpu([reduced_union])
            total_correct.update(reduced_correct.item(), 1)
            total_label.update(reduced_label.item(), 1)
            #total_inter.update(reduced_inter.item(), image.size(0))
            #total_union.update(reduced_union.item(), image.size(0))

            #total_correct += correct
            #total_label += labeled
            #total_inter += inter
            #total_union += union
            total_inter += reduced_inter
            total_union += reduced_union
            pixAcc = 1.0 * total_correct.sum / (np.spacing(1) + total_label.sum)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            if self.args.rank == 0:
                tbar.set_description(
                    'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

        new_pred = (pixAcc + mIoU)/2
        self.track_pixAcc = pixAcc
        self.track_mIoU = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if self.args.rank == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                #'state_dict': self.model.module.state_dict(),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)


if __name__ == "__main__":
    args = Options().parse()
    #args.add_argument("--local_rank", type=int, default=0)
    #torch.multiprocessing.set_start_method('spawn')
    seed = args.seed
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.enabled=True
    #world_size = init_dist_slurm(args)
    #args.local_rank = dist.get_rank()
    #args.world_size = world_size
    args.local_rank = init_dist_slurm(args)
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    seed = args.rank
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    trainer = Trainer(args)
    if args.rank == 0:
        directory = "runs/summary/%s/%s/%s/"%(args.dataset, args.model, args.checkname)
        if not os.path.exists(directory):
            os.makedirs(directory)
        writer = SummaryWriter(directory)
        print('Starting Epoch:', trainer.args.start_epoch)
        print('Total Epoches:', trainer.args.epochs)
    if args.eval:
        trainer.validation(trainer.args.start_epoch)
    else:
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                trainer.validation(epoch)
            if args.rank == 0:
                writer.add_scalar('loss', trainer.track_loss, epoch)
                writer.add_scalar('best_pred', trainer.best_pred, epoch)
                writer.add_scalar('pixAcc', trainer.track_pixAcc, epoch)
                writer.add_scalar('mIoU', trainer.track_mIoU, epoch)

    if args.rank == 0:
        writer.close()
