#!/bin/bash
srun -p XXX --gres=gpu:8 --job-name=net_test3 \
python test.py --dataset ade20k --model efficientFCN  --backbone resnet101 --backbone-path $1 --num-center 600 --base-size 608 --crop-size 576 --ms --resume $2 --eval

