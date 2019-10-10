#!/bin/bash
mkdir /tandatasets/lsadam-experiments-radam-repo/vanilla-adam
# ln -s /nvdatasets/imagenet/train/ ./lsadam/train
# ln -s /nvdatasets/imagenet/val/ ./lsadam/val
# mkdir /tandatasets/lsadam-experiments/ls-adam
python imagenet.py -j 16 -a resnet18 --data /nvdatasets/imagenet/ --epochs 90 --schedule 31 61 --gamma 0.1 -c /tandatasets/lsadam-experiments-radam-repo/vanilla-adam --model_name adam_01 --optimizer adam --lr 0.01 --beta1 0.9 --beta2 0.999 --gpu-id '0,1,2,3,4,5,6,7'