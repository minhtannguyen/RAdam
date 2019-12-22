#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for ((i=0;i<2;++i))
do
    arch_name="resnet18"
    gpu_id="0,1,2,3,4,5,6,7"
    optimizer_name="sradamw"
    wd_value=5e-4
    n_value=45

    mkdir /tanresults/experiments-restarting/imagenet-${arch_name}-wd-${wd_value}-${optimizer_name}-lr-0-01-gamma-0-1-scheme1-n-${n_value}-seed-${i}

    python imagenet.py -j 16 -a ${arch_name} --data /nvdatasets/imagenet/ --epochs 90 --schedule 31 61 --restart-schedule 45 90 180 --gamma 0.1 --lr 0.01 -c /tanresults/experiments-restarting/imagenet-${arch_name}-wd-${wd_value}-${optimizer_name}-lr-0-01-gamma-0-1-scheme1-n-${n_value}-seed-${i} --model_name ${arch_name}_sradam --optimizer ${optimizer_name} --gpu-id ${gpu_id} --manualSeed ${i}
done
