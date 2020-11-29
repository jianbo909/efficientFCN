#!/bin/bash
srun -p XXX --gres=gpu:8 --job-name=net_test3 \
python test.py --dataset PContext --model efficientFCN  --backbone resnet101 --backbone-path $1 --crop-size 512 --lateral --ms --resume $2 --eval

