#!/bin/bash
python list_gen.py --img_dir=./image --list_num=1


CUDA_VISIBLE_DEVICES=0 python run_pipeline.py 0 1>log_tmp0 2>&1 &
sleep 20
# Data Procession with multiple GPUs
# CUDA_VISIBLE_DEVICES=1 python run_pipeline.py 1 1>log_tmp1 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=2 python run_pipeline.py 2 1>log_tmp2 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=3 python run_pipeline.py 3 1>log_tmp3 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=4 python run_pipeline.py 4 1>log_tmp4 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=5 python run_pipeline.py 5 1>log_tmp5 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=6 python run_pipeline.py 6 1>log_tmp6 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=7 python run_pipeline.py 7 1>log_tmp7 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=0 python run_pipeline.py 8 1>log_tmp8 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=1 python run_pipeline.py 9 1>log_tmp9 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=2 python run_pipeline.py 10 1>log_tmp10 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=3 python run_pipeline.py 11 1>log_tmp11 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=4 python run_pipeline.py 12 1>log_tmp12 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=5 python run_pipeline.py 13 1>log_tmp13 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=6 python run_pipeline.py 14 1>log_tmp14 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=7 python run_pipeline.py 15 1>log_tmp15 2>&1 &
# sleep 20

