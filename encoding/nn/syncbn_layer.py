import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.distributed as dist
import torch.nn as nn

class SyncBNFunc(Function):

    @staticmethod
    def forward(ctx, in_data, scale_data, shift_data, running_mean, running_var, eps, momentum, training):
        if in_data.is_cuda:
            ctx.eps =eps
            N, C, H, W = in_data.size()
            in_data = in_data.view(N, C, -1)
            mean_in = in_data.mean(-1, keepdim=True)
            #var_in = in_data.var(-1, keepdim=True)
            var_in = in_data.var(-1, keepdim=True, unbiased=False)
            num_pixel = H * W
            alpha = num_pixel / (num_pixel - 1 + ctx.eps)
            var_in = var_in * alpha
            temp = var_in + mean_in ** 2
            scale_data = scale_data.view(1, C, 1, 1)
            shift_data = shift_data.view(1, C, 1, 1)
            running_mean = running_mean.view(1, C, 1)
            running_var = running_var.view(1, C, 1)
            if training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2

                sum_x = mean_bn ** 2 + var_bn
                dist.all_reduce(mean_bn)
                mean_bn /= dist.get_world_size()
                dist.all_reduce(sum_x)
                sum_x /= dist.get_world_size()
                var_bn = sum_x - mean_bn ** 2

                running_mean.mul_(momentum)
                running_mean.add_((1 - momentum) * mean_bn.data)
                running_var.mul_(momentum)
                running_var.add_((1 - momentum) * var_bn.data)

                #print('SyncBN: ws: %.3f, eps: %.8f' % (dist.get_world_size(), ctx.eps))

            else:
                mean_bn = torch.autograd.Variable(running_mean)
                var_bn = torch.autograd.Variable(running_var)

            x_hat = (in_data - mean_bn) / (var_bn+ ctx.eps).sqrt()
            x_hat = x_hat.view(N, C, H, W)
            out_data = x_hat * scale_data + shift_data

            #print('SyncBN: xhat_max: %.3f, xhat_min: %.3f, scale_max: %.3f, scale_min: %.3f, shift_max:%.3f, shift_min:%.3f, out_max:%.3f, out_min:%.3f' % (torch.max(x_hat).item(), torch.min(x_hat).item(), torch.max(scale_data).item(), torch.min(scale_data).item(), torch.max(shift_data).item(), torch.min(shift_data).item(), torch.max(out_data).item(), torch.min(out_data).item()))

            ctx.save_for_backward(in_data.data, scale_data.data, x_hat.data,  mean_bn.data, var_bn.data)
            #scale_data = scale_data.view(C)
            #shift_data = shift_data.view(C)
            #running_mean = running_mean.view(C)
            #running_var = running_var.view(C)
        else:
            raise RuntimeError('SyncBNFunc only support CUDA computation!')
        return out_data

    @staticmethod
    def backward(ctx, grad_outdata):
        if grad_outdata.is_cuda:

            in_data, scale_data, x_hat, mean_bn, var_bn =  ctx.saved_tensors

            N, C, H, W = grad_outdata.size()
            scaleDiff = torch.sum(grad_outdata * x_hat,[0,2,3],keepdim=True)
            shiftDiff = torch.sum(grad_outdata,[0,2,3],keepdim=True)
            dist.all_reduce(scaleDiff)
            dist.all_reduce(shiftDiff)

            inDiff = scale_data / (var_bn.view(1,C,1,1) + ctx.eps).sqrt() *(grad_outdata - 1 / (N*H*W*dist.get_world_size()) * (scaleDiff * x_hat + shiftDiff))
            scaleDiff = scaleDiff.view(C)
            shiftDiff = shiftDiff.view(C)

            #print('SyncBN: grad_out_max: %.3f, grad_out_min: %.3f, eps: %.8f' % (torch.max(grad_outdata).item(), torch.min(grad_outdata).item(), ctx.eps))

        else:
            raise RuntimeError('SyncBNFunc only support CUDA computation!')
        return inDiff, scaleDiff, shiftDiff, None, None, None, None, None

class SyncBatchNorm2d(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.9,last_gamma=False):
        super(SyncBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.last_gamma = last_gamma

        #self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        #self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))

        #self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        #self.register_buffer('running_var', torch.ones(1, num_features, 1))

        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))


        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum})'
                .format(name=self.__class__.__name__, **self.__dict__))

    def forward(self, in_data):
        return SyncBNFunc.apply(
                    in_data, self.weight, self.bias,  self.running_mean, self.running_var, self.eps, self.momentum, self.training)
