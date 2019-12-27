#!/bin/bash

arch_name='hopreresnet'
block_name="basicblock"
out_type='x'
dataset_name="cifar10"
eta_value=3
depth_value=20
gpu_id=0
optimizer_name='sgd'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for ((i=0;i<1;++i))
do
    mkdir /tanresults/experiments-horesnet/${dataset_name}-${arch_name}${depth_value}-${block_name}-eta-${eta_value}-${out_type}-opt-${optimizer_name}-seed-${i}
    python cifar_horesnet_v4.py -a ${arch_name} --block-name ${block_name} --dataset ${dataset_name} --depth ${depth_value} --epochs 200 --schedule 80 120 160 --gamma 0.1 --lr 0.1 --wd 5e-4 --checkpoint /tanresults/experiments-horesnet/${dataset_name}-${arch_name}${depth_value}-${block_name}-eta-${eta_value}-${out_type}-opt-${optimizer_name}-seed-${i} --gpu-id ${gpu_id} --model_name ${arch_name}20_sgd_1 --eta ${eta_value} --feature_vec ${out_type} --optimizer ${optimizer_name} --manualSeed ${i}
done