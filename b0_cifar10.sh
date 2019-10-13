#!/bin/bash
# 
#SBATCH --job-name=effnet_cifar10_1
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/jlli/logs/effnet_cifar10.out

module load cuda/10.1
module load cudnn/7.6.2

PYTHON=/home/jlli/anaconda3/envs/ml/bin/python
MAIN=/home/jlli/convNet.pytorch/main.py
DATA=/home/jlli/datasets
OUTPUT=/home/jlli/convNet.pytorch/cifar10_results
SEED=1
LR=0.1
BATCH=128
DECAY=5e-4

$PYTHON $MAIN \
	--results-dir OUTPUT \
	--datasets-dir $DATA \
	--dataset cifar10 \
	--model efficientnet \
	--model-config  '{"resolution": 32, "num_classes": 10, "use_cifar": True}' \
	--dtype float \
	--device cuda \
	--epochs 200 \
	--batch-size $BATCH \
	--lr $LR \
	--weight-decay $DECAY \
	--print-freq 10 \
	--seed $SEED
	
