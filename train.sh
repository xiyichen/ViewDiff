module load stack/2024-04 cuda/11.8.0 gcc/8.5.0 eth_proxy ffmpeg/6.0
# ./viewdiff/scripts/train_small.sh ./shapenet "stabilityai/stable-diffusion-2-1-base" outputs/train category=teddybear
# ./viewdiff/scripts/train_small.sh ./shapenet "./stable-diffusion-2-1-base" outputs/train category=teddybear
# export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN

# ./viewdiff/scripts/train.sh ../co3d/ "./stable-diffusion-2-1-base" outputs/train category=teddybear


# ./viewdiff/scripts/train.sh ../co3d/ "./stable-diffusion-2-1-base" outputs_full/train category=teddybear

./viewdiff/scripts/train.sh ../co3d/ "./stable-diffusion-2-1-base" outputs_full/train category=teddybear
