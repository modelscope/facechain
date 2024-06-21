#!/bin/bash
# # ------------------------example--------------------------

# reg data:
CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 10 --n_iter 5 --scale 10.0 --ddim_steps 50  --ckpt stable-diffusion-v-1-4/sd-v1-4-full-ema.ckpt --prompt "photo of a dog" --seed 77 --outdir reg_data/dog --unconditional_prompt "monochrome, lowres, bad anatomy, worst quality, low quality"

# customize
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml -t --actual_resume stable-diffusion-v-1-4/sd-v1-4-full-ema.ckpt -n task_dog --gpus 0, --data_root examples/dog5 --reg_data_root reg_data/dog/samples --class_word "dog" --logdir output_checkpoints --sude_weight 1.6

# generate:
CUDA_VISIBLE_DEVICES=0 python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 2 --scale 10.0 --ddim_steps 50  --ckpt output_checkpoints/xxxxxxxxxxxx/checkpoints/epoch=000004.ckpt --prompt "photo of a running sks dog" --seed 77 --outdir output_images --unconditional_prompt "monochrome, lowres, bad anatomy, worst quality, low quality"

