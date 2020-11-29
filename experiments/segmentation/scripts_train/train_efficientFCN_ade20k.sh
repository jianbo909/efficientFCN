#!/bin/bash
#CUDA_VISBILE_DEVICES=0,1,2,3 \
srun -p XXX \
     --nodes=2 --gres=gpu:8 --job-name=test2_t2 \
     --ntasks=16 --ntasks-per-node=8\
     --kill-on-bad-exit=1 \
python -m torch.distributed.launch train_dist.py --dataset ade20k --model efficientFCN --backbone resnet101 --backbone-path $1 --num-center 600 --batch-size 32 --test-batch-size 32 --crop-size 576 --workers 16 --lr 0.002 --epochs 80 --lateral --checkname efficientFCN_resnet101_ade20k
