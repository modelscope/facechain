#!/bin/bash
export IMAGE_ROOT="../data_process/cropimg"
export FACE_ROOT="../data_process/aligned_masked"
export CAPTION_ROOT="../data_process/caption"
export JOB_NAME="job_name"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_maj.py  --IMAGE_ROOT=$IMAGE_ROOT --FACE_ROOT=$FACE_ROOT --CAPTION_ROOT=$CAPTION_ROOT --yaml_file=configs/mirror_wofc.yaml --batch_size=3 --name=$JOB_NAME --face_prob=0.1  
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_film.py  --IMAGE_ROOT=$IMAGE_ROOT --FACE_ROOT=$FACE_ROOT --CAPTION_ROOT=$CAPTION_ROOT --yaml_file=configs/mirror.yaml --batch_size=3 --name=$JOB_NAME --face_prob=0.1  