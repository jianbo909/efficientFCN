#!/bin/bash
srun -p XXXX \
     --nodes=1 --gres=gpu:8 --job-name=test2_enc_t2 \
     --ntasks=8 --ntasks-per-node=8\
     --kill-on-bad-exit=1 \
python -m torch.distributed.launch train_dist.py --dataset PContext --model efficientFCN --backbone resnet101 --backbone-path $1 --batch-size 16 --test-batch-size 32 --crop-size 512 --workers 1 --lr 0.001 --epochs 120 --checkname efficientFCN
