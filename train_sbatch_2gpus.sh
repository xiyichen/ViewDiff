#!/bin/bash
#SBATCH --gpus=rtx_3090:2
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=120:00:00
#SBATCH --output=./output_2gpus.txt
#SBATCH --error=./output_2gpus.txt
#SBATCH --cpus-per-task=16
#SBATCH --job-name=viewdiff_2gpus

export NCCL_DEBUG=INFO

module load stack/2024-04 cuda/11.8.0 gcc/8.5.0 eth_proxy ffmpeg/6.0
export NCCL_DEBUG=WARN

./viewdiff/scripts/train.sh ../co3d/ "./stable-diffusion-2-1-base" outputs/train category=teddybear
