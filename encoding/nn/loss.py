import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
__all__ = ['SegmentationLosses', 'SegmentationLosses2', 'SegmentationLosses3', 'debug_loss1_SegmentationLosses', 'debug_loss2_SegmentationLosses', 'debug_loss3_SegmentationLosses', 'OhemCrossEntropy2d', 'OHEMSegmentationLosses']

class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 

    def forward(self, inputs, targets):
        #preds, target = tuple(inputs)
        #inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            #return super(SegmentationLosses, self).forward(*inputs)
            preds = list(inputs)
            loss = super(SegmentationLosses, self).forward(preds[0], targets)
            return loss
        elif not self.se_loss:
            #pred1, pred2, target = tuple(inputs)
            preds = list(inputs)
            pred1 = preds[0]
            pred2 = preds[1]
            target = targets
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            #return loss1 + self.aux_weight * loss2
            loss = loss1 + self.aux_weight * loss2
            return loss
        elif not self.aux:
            #pred, se_pred, target = tuple(inputs)
            preds = list(inputs)
            pred = preds[0]
            se_pred = preds[1]
            target = targets
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            if (se_pred.size() == se_target.size()):
                loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            else:
                loss3 = super(SegmentationLosses, self).forward(se_pred, target)
            #print("loss_pred=%.3f, loss_aux=%.3f, loss_se=%.3f" % (loss1, loss2, loss3))
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

class SegmentationLosses2(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(SegmentationLosses2, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses2, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses2, self).forward(pred1, target)
            loss2 = super(SegmentationLosses2, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses2, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred0_5, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses2, self).forward(pred1, target)
            loss0_5 = super(SegmentationLosses2, self).forward(pred0_5, target)
            loss2 = super(SegmentationLosses2, self).forward(pred2, target)
            if (se_pred.size() == se_target.size()):
                loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            else:
                loss3 = super(SegmentationLosses2, self).forward(se_pred, target)
            #print("loss_pred=%.3f, loss_aux=%.3f, loss_se=%.3f" % (loss1, loss2, loss3))
            return loss1 + self.aux_weight * loss0_5 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

class SegmentationLosses3(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(SegmentationLosses3, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        #self.bceloss = nn.BCELoss(weight) 

    def forward(self, inputs, targets):
        preds = list(inputs)
        pred1 = preds[0]
        pred0_1 = preds[1]
        pred0_2 = preds[2]
        target = targets
        #pred1, pred0_1,  pred0_2,  pred0_3, pred2, target = tuple(inputs)
        #se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
        loss1 = super(SegmentationLosses3, self).forward(pred1, target)
        loss0_1 = super(SegmentationLosses3, self).forward(pred0_1, target)
        loss0_2 = super(SegmentationLosses3, self).forward(pred0_2, target)
        #loss0_3 = super(SegmentationLosses3, self).forward(pred0_3, target)
        #loss2 = super(SegmentationLosses3, self).forward(pred2, target)
        #print("loss_pred=%.3f, loss_aux=%.3f, loss_se=%.3f" % (loss1, loss2, loss3))
        #return loss1 + 0.3 * loss0_1 + 0.3 * loss0_2 + 0.3 * loss0_3 + self.aux_weight * loss2 
        return loss1 + self.aux_weight * loss0_1 + self.aux_weight * loss0_2

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect




class debug_loss1_SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(debug_loss1_SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 

    def forward(self, *inputs):
        pred1, se_pred, pred2, target = tuple(inputs)
        loss1 = super(debug_loss1_SegmentationLosses, self).forward(pred1, target)
        return loss1

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

class debug_loss2_SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(debug_loss2_SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 

    def forward(self, *inputs):
        pred1, se_pred, pred2, target = tuple(inputs)
        loss2 = super(debug_loss2_SegmentationLosses, self).forward(pred2, target)
        return loss2

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect


class debug_loss3_SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(debug_loss3_SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 

    def forward(self, *inputs):
        pred1, se_pred, pred2, target = tuple(inputs)
        se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
        loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
        return loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

#class OhemCrossEntropy(nn.Module): 
class OhemCrossEntropy2d(nn.Module):
    #def __init__(self, ignore_label=-1, thres=0.9, 
    #    min_kept=131072): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000): 
        super(OhemCrossEntropy2d, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                1.0865, 1.1529, 1.0507]).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        if pred.numel() > 1:
            min_value = pred[min(self.min_kept, pred.numel() - 1)] 
            threshold = max(min_value, self.thresh) 
        
            pixel_losses = pixel_losses[mask][ind]
            pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()



class OHEMSegmentationLosses(OhemCrossEntropy2d):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 thres=0.7, min_kept=100000,
                 ignore_index=-1):
        super(OHEMSegmentationLosses, self).__init__(ignore_label=ignore_index, thres=thres, min_kept=min_kept)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 

    #def forward(self, *inputs):
    def forward(self, inputs, targets):
        if not self.se_loss and not self.aux:
            preds = list(inputs)
            pred1 = preds[0]
            target = targets
            #return super(OHEMSegmentationLosses, self).forward(*inputs)
            return super(OHEMSegmentationLosses, self).forward(pred1, target)
        elif not self.se_loss:
            #pred1, pred2, target = tuple(inputs)
            preds = list(inputs)
            pred1 = preds[0]
            pred2 = preds[1]
            target = targets
            loss1 = super(OHEMSegmentationLosses, self).forward(pred1, target)
            loss2 = super(OHEMSegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(OHEMSegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(OHEMSegmentationLosses, self).forward(pred1, target)
            loss2 = super(OHEMSegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect
